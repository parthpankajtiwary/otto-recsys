import time, pickle, gc

import numpy as np
import pandas as pd

from tqdm import tqdm

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

from collections import defaultdict, Counter
import cudf, itertools

from annoy import AnnoyIndex
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')

class config:
    data_path = 'data/'
    local_validation = False
    debug = False
    word2vec = True
    validation_path = 'data/local_validation/'
    train_file = 'train.parquet'
    test_file = 'test.parquet'
    test_labels_file = 'test_labels.parquet'
    submission_path = 'submissions/'
    submission_file = 'submission_{:%Y-%m-%d_%H-%M}.csv'
    model_path = 'models/word2vec.model'
    type_labels = {'clicks':0, 'carts':1, 'orders':2}
    type_weight = {0:1, 1:6, 2:3}
    version = 1
    chunk_size = 100_000
    random_state = 42
    fraction = 0.02
    n_samples = 30
    n_top = 15
    diff_clicks = 24 * 60 * 60


def load_data():
    """Load data from parquet files."""
    if config.local_validation:
        train = cudf.read_parquet(config.validation_path + config.train_file)
        test = cudf.read_parquet(config.validation_path + config.test_file)
        data = cudf.concat([train, test])
    else:
        train = cudf.read_parquet(config.data_path + config.train_file)
        test = cudf.read_parquet(config.data_path + config.test_file)
        data = cudf.concat([train, test])    
    if config.debug:
        data = data.sample(frac=config.fraction, random_state=config.random_state)
    return data, test


def load_model():
    """Load word2vec model."""
    if config.word2vec:
        print('Loading word2vec model...')
        model = Word2Vec.load(config.model_path)
        return model
    else:
        return None


def build_index(model, n_trees=100):
    """Build index for word2vec model."""
    if config.word2vec:
        print('Building index for word2vec model...')
        aid2idx = {aid: i for i, aid in enumerate(model.wv.index_to_key)}
        index = AnnoyIndex(model.wv.vector_size, metric='euclidean')
        for aid, idx in aid2idx.items():
            index.add_item(idx, model.wv.vectors[idx])
        index.build(n_trees=n_trees)
        return index, aid2idx
    else:
        return None, None


def process_covisitation_in_chunks(data, chunk_size):
    """Process data in chunks and return a dataframe."""
    tmp = list()
    data = data.set_index('session')
    sessions = data.index.unique()

    for i in tqdm(range(0, sessions.shape[0], chunk_size)):
        df = data.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index()
        df = df.sort_values(['session','ts'],ascending=[True, False])
        df = df.reset_index(drop=True)

        df['n'] = df.groupby('session').cumcount()
        df = df.loc[df.n<config.n_samples].drop('n', axis=1)

        df = df.merge(df,on='session')
        df = df.loc[ ((df.ts_x - df.ts_y).abs() < config.diff_clicks) & (df.aid_x != df.aid_y) ]

        df = df[['session', 'aid_x', 'aid_y','ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
        df['wgt'] = 1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800)
        df = df[['aid_x','aid_y','wgt']]
        df.wgt = df.wgt.astype('float32')
        df = df.groupby(['aid_x','aid_y']).wgt.sum()

        tmp.append(df.reset_index())
        
        del df
        gc.collect()

    return tmp 


def combine_covisitation_chunks(tmp):
    """Combine covisitation chunks and return a dataframe."""
    tmp = list(map(lambda x: x.to_pandas(), tmp))
    tmp = pd.concat(tmp)
    return tmp


def generate_combined_covisitation(data, chunk_size):
    """Generate combined covisitation."""
    print('Processing co-visitation matrix in chunks...')
    tmp = process_covisitation_in_chunks(data, chunk_size)
    tmp = combine_covisitation_chunks(tmp)
    print('Generating combined covisitation matrix...')
    tmp = tmp.groupby(['aid_x','aid_y']).wgt.sum().reset_index()
    tmp = tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    # we only select n products for each aid_x
    tmp = tmp.loc[tmp.n<config.n_top].drop('n',axis=1)
    df = tmp.groupby('aid_x').aid_y.apply(list)
    save_combined_covisitation(df)
    print('Combined covisitation matrix saved.')


def save_combined_covisitation(df):
    """Save combined covisitation."""
    with open(config.data_path + f'top_20_v{config.version}.pkl', 'wb') as f:
        pickle.dump(df.to_dict(), f)


def load_combined_covisitation(test, version=config.version):
    top_20 = pd.read_pickle(config.data_path + f'top_20_v{version}.pkl')

    # TOP CLICKS AND ORDERS IN TEST
    top_clicks = test.loc[test['type']==config.type_labels['clicks'],'aid'] \
                          .value_counts().index.values[:20]
    top_orders = test.loc[test['type']==config.type_labels['orders'],'aid'] \
                          .value_counts().index.values[:20]

    print('Size of top_20_clicks:', len(top_20))

    return top_20, top_clicks, top_orders


def get_nns(model, index, product, aid2idx, n=21):
    """Get nearest neighbors for a given aid."""
    idx = aid2idx[product]
    nns = index.get_nns_by_item(idx, n)[1:]
    nns = [model.wv.index_to_key[idx] for idx in nns]
    return nns


def suggest_clicks(df, top_20, top_clicks, model, index, aid2idx):
    products = df.aid.tolist()
    types = df.type.tolist()
    unique_products = list(dict.fromkeys(products[::-1] ))

    if len(unique_products) >= 20:
        weights = np.logspace(0.1, 1, len(products), base=2, endpoint=True) - 1
        products_tmp = Counter()

        for product, weight, _type in zip(products, weights, types):
            products_tmp[product] += weight * config.type_weight[_type]
        
        sorted_products = [product for product, _ in products_tmp.most_common(50)]
        return sorted_products

    products_1 = list(itertools.chain(*[top_20[product] \
                    for product in unique_products if product in top_20]))

    if config.word2vec:
        products = list(dict.fromkeys(products[::-1]))
        # most_recent_product = products[0]
        # nns = [model.wv.index_to_key[i]
        #        for i in index.get_nns_by_item(aid2idx[most_recent_product], 21)[1:]]
        products_2 = list(itertools.chain(*[get_nns(model, index, product, aid2idx) 
                          for product in unique_products if product in top_20]))


        top_products_1 = [product for product, _ in Counter(products_1 + products_2).most_common(50) \
                        if product not in unique_products]
    else:
        top_products_1 = [product for product, _ in Counter(products_1).most_common(50) \
                        if product not in unique_products]

    result = unique_products + top_products_1[:20 - len(unique_products)]
    return result + list(top_clicks[:20 - len(result)])


def generate_candidates(test, top_20, top_clicks, model, index, aid2idx):
    test = test.to_pandas()

    tqdm.pandas()

    pred_df = test.sort_values(["session", "ts"]) \
                  .groupby(["session"]) \
                  .progress_apply(lambda x: suggest_clicks(x, top_20, top_clicks, model, index, aid2idx)) 

    clicks_pred_df = pd.DataFrame(pred_df.add_suffix("_clicks"), columns=["labels"]).reset_index()
    orders_pred_df = pd.DataFrame(pred_df.add_suffix("_orders"), columns=["labels"]).reset_index()
    carts_pred_df = pd.DataFrame(pred_df.add_suffix("_carts"), columns=["labels"]).reset_index()

    pred_df = pd.concat([clicks_pred_df, orders_pred_df, carts_pred_df])
    pred_df.columns = ["session_type", "labels"]
    pred_df["labels"] = pred_df.labels.progress_apply(lambda x: " ".join(map(str,x)))

    if not config.local_validation:
        pred_df.to_csv(config.submission_path + config.submission_file, index=False)

    return pred_df


def compute_validation_score(pred):
    if config.local_validation:
        score = 0
        weights = {'clicks': 0.10, 'carts': 0.30, 'orders': 0.60}
        for t in ['clicks','carts','orders']:
            sub = pred.loc[pred.session_type.str.contains(t)].copy()
            sub['session'] = sub.session_type.apply(lambda x: int(x.split('_')[0]))
            sub.labels = sub.labels.apply(lambda x: [int(i) for i in x.split(' ')[:20]])
            test_labels = pd.read_parquet(config.validation_path + config.test_labels_file)
            test_labels = test_labels.loc[test_labels['type']==t]
            test_labels = test_labels.merge(sub, how='left', on=['session'])
            test_labels['hits'] = test_labels.apply(lambda df: len(set(df.ground_truth) \
                                             .intersection(set(df.labels))), axis=1)
            test_labels['gt_count'] = test_labels.ground_truth.str.len().clip(0,20)
            recall = test_labels['hits'].sum() / test_labels['gt_count'].sum()
            score += weights[t]*recall
            print(f'{t} recall =', recall)
            
        print('=============')
        print('Overall Recall =',score)
        print('=============')


"""Main module."""
def main():
    data, test = load_data()
    # generate_combined_covisitation(data, config.chunk_size)
    top_20, top_clicks, top_orders = load_combined_covisitation(test)
    word2vec = load_model()
    index, aid2idx = build_index(word2vec)
    pred = generate_candidates(test, top_20, top_clicks, word2vec, index, aid2idx)
    compute_validation_score(pred)

if __name__ == '__main__':
    main()