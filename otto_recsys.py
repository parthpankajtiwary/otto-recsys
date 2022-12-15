import time, pickle, gc

import numpy as np
import pandas as pd

from tqdm import tqdm

from collections import Counter
import cudf, itertools

from annoy import AnnoyIndex
from gensim.models import Word2Vec

import warnings
warnings.filterwarnings('ignore')

from utils.submit import submit_file

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
    submission_file = 'submission_{0}.csv'.format(time.strftime("%Y%m%d-%H%M%S"))    
    model_path = 'models/word2vec-windowsize-20-full-data.model'
    type_labels = {'clicks':0, 'carts':1, 'orders':2}
    type_weight = {0:1, 1:6, 2:3}
    version = 1
    chunk_size = 100_000
    random_state = 42
    fraction = 0.002
    n_top = 15
    n_samples = 30
    time_diff = 24 * 60 * 60


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


def get_nns(model, index, product, aid2idx, n=21):
    """Get nearest neighbors for a given aid."""
    try:
        idx = aid2idx[product]
        nns = index.get_nns_by_item(idx, n)[1:]
        nns = [model.wv.index_to_key[idx] for idx in nns]
    except:
        nns = []
    return nns


def process_covisitation_in_chunks(data, time_diff, chunk_size, type='clicks'):
    """Process data in chunks and return a dataframe."""
    tmp = list()
    data = data.set_index('session')
    sessions = data.index.unique()

    for i in tqdm(range(0, sessions.shape[0], chunk_size)):
        df = data.loc[sessions[i]:sessions[min(sessions.shape[0]-1, i+chunk_size-1)]].reset_index()
        if type == 'buy2buy':
            df = df.loc[(df['type']==1)|(df['type']==2)]
        df = df.sort_values(['session','ts'],ascending=[True, False])
        df = df.reset_index(drop=True)

        df['n'] = df.groupby('session').cumcount()
        df = df.loc[df.n<config.n_samples].drop('n', axis=1)

        df = df.merge(df,on='session')
        df = df.loc[ ((df.ts_x - df.ts_y).abs() < time_diff) & (df.aid_x != df.aid_y) ]

        if type == 'clicks':
            df = df[['session', 'aid_x', 'aid_y','ts_x']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = 1 + 3*(df.ts_x - 1659304800)/(1662328791-1659304800)
        if type == 'carts-orders':
            df = df[['session', 'aid_x', 'aid_y','type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = df.type_y.map(config.type_weight)
        if type == 'buy2buy':
            df = df[['session', 'aid_x', 'aid_y','type_y']].drop_duplicates(['session', 'aid_x', 'aid_y'])
            df['wgt'] = 1

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


def generate_combined_covisitation(data, chunk_size, type='clicks', time_diff=config.time_diff):
    """Generate combined covisitation."""
    print('Processing co-visitation matrix in chunks...')
    tmp = process_covisitation_in_chunks(data, time_diff, chunk_size, type)
    tmp = combine_covisitation_chunks(tmp)
    print('Generating combined covisitation matrix...')
    tmp = tmp.groupby(['aid_x','aid_y']).wgt.sum().reset_index()
    tmp = tmp.sort_values(['aid_x','wgt'],ascending=[True,False])
    tmp = tmp.reset_index(drop=True)
    tmp['n'] = tmp.groupby('aid_x').aid_y.cumcount()
    # we only select n products for each aid_x
    tmp = tmp.loc[tmp.n<config.n_top].drop('n',axis=1)
    df = tmp.groupby('aid_x').aid_y.apply(list)
    save_combined_covisitation(df, type)
    print('Combined covisitation matrix saved.')
    del tmp, df


def save_combined_covisitation(df, type='clicks'):
    """Save combined covisitation."""
    with open(config.data_path + f'top_20_{type}_v{config.version}.pkl', 'wb') as f:
        pickle.dump(df.to_dict(), f)


def get_top_clicks_orders(df):
    top_clicks = df.loc[df['type']==config.type_labels['clicks'],'aid'] \
                          .value_counts().index.values[:20]
    top_orders = df.loc[df['type']==config.type_labels['orders'],'aid'] \
                          .value_counts().index.values[:20]
    return top_clicks, top_orders


def load_combined_covisitation(version=config.version, type='clicks'):
    top_20 = pd.read_pickle(config.data_path + f'top_20_{type}_v{version}.pkl')
    print(f'Size of top_20_{type}:', len(top_20))
    return top_20


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
        products_2 = list(itertools.chain(*[get_nns(model, index, product, aid2idx) 
                          for product in unique_products if product in top_20]))
        top_word2vec = [product for product, _ in Counter(products_2).most_common(50) \
                        if product not in unique_products]

        top_products = [product for product, _ in Counter(products_1).most_common(50) \
                        if product not in unique_products]
        result = unique_products + top_products[:20 - len(unique_products)]
        return result + list(top_word2vec[:20 - len(result)])
    else:
        top_products = [product for product, _ in Counter(products_1).most_common(50) \
                        if product not in unique_products]

        result = unique_products + top_products[:20 - len(unique_products)]
        return result + list(top_clicks[:20 - len(result)])


def suggest_carts(df, top_15_buy2buy, top_15_buys, top_orders,
                 model, index, aid2idx):
    products = df.aid.tolist()
    types = df.type.tolist()
    # filter df for type 1 and 2
    unique_products = list(dict.fromkeys(products[::-1] ))
    df = df.loc[(df['type']==1)|(df['type']==2)]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))

    if len(unique_products) >= 20:
        weights = np.logspace(0.5, 1, len(products), base=2, endpoint=True) - 1
        products_tmp = Counter()
        for product, weight, _type in zip(products, weights, types):
            products_tmp[product] += weight * config.type_weight[_type]
        products_1 = list(itertools.chain(*[top_15_buy2buy.get(product, []) \
                        for product in unique_buys if product in top_15_buy2buy]))
        for product in products_1: products_tmp[product] += 0.1
        sorted_products = [product for product, _ in products_tmp.most_common(50)]
        return sorted_products

    products_1 = list(itertools.chain(*[top_15_buys.get(product, []) \
                        for product in unique_products if product in top_15_buys]))
    products_2 = list(itertools.chain(*[top_15_buy2buy.get(product, []) \
                        for product in unique_buys if product in top_15_buy2buy]))
    top_products = [product for product, _ in Counter(products_1 + products_2).most_common(50) \
                    if product not in unique_products]
    result = unique_products + top_products[:20 - len(unique_products)]

    if config.word2vec:
        products_3 = list(itertools.chain(*[get_nns(model, index, product, aid2idx) 
                          for product in unique_products if product in top_15_buys]))
        top_word2vec = [product for product, _ in Counter(products_3).most_common(50) \
                        if product not in unique_products]
        return result + list(top_word2vec[:20 - len(result)])

    return result + list(top_orders[:20 - len(result)])


def suggest_orders(df, top_15_buy2buy, top_15_buys, top_orders,
                   model, index, aid2idx):
    products = df.aid.tolist()
    types = df.type.tolist()
    # filter df for type 1 and 2
    unique_products = list(dict.fromkeys(products[::-1] ))
    df = df.loc[(df['type']==1)|(df['type']==2)]
    unique_buys = list(dict.fromkeys(df.aid.tolist()[::-1]))
    added_to_cart = list(dict.fromkeys(df.loc[df['type']==1].aid.tolist()[::-1]))

    if len(unique_products) >= 20:
        weights = np.logspace(0.5, 1, len(products), base=2, endpoint=True) - 1
        products_tmp = Counter()
        for product, weight, _type in zip(products, weights, types):
            products_tmp[product] += weight * config.type_weight[_type]
        products_1 = list(itertools.chain(*[top_15_buy2buy.get(product, []) \
                          for product in unique_buys if product in top_15_buy2buy]))
        for product in products_1: products_tmp[product] += 0.1
        sorted_products = [product for product, _ in products_tmp.most_common(50)]
        return sorted_products

    products_1 = list(itertools.chain(*[top_15_buys.get(product, []) \
                        for product in unique_products if product in top_15_buys]))
    products_2 = list(itertools.chain(*[top_15_buy2buy.get(product, []) \
                        for product in unique_buys if product in top_15_buy2buy]))
    top_products = [product for product, _ in Counter(products_1 + products_2).most_common(50) \
                    if product not in unique_products]

    if added_to_cart:
        added_to_cart = [product for product in added_to_cart \
                         if product not in unique_products]
        result = unique_products + added_to_cart + top_products[:20 - len(unique_products)]
    else:
        result = unique_products + top_products[:20 - len(unique_products)]

    if config.word2vec:
        products_3 = list(itertools.chain(*[get_nns(model, index, product, aid2idx) 
                          for product in unique_products if product in top_15_buys]))
        top_word2vec = [product for product, _ in Counter(products_3).most_common(50) \
                        if product not in unique_products]
        return result + list(top_word2vec[:20 - len(result)])

    return result + list(top_orders[:20 - len(result)])


def generate_candidates(test, covisit_clicks, covisit_carts_orders, 
                        covisits_buy2buy, top_clicks, top_orders, 
                        model, index, aid2idx):
    test = test.to_pandas()

    tqdm.pandas()

    pred_df_clicks = test.sort_values(["session", "ts"]) \
                         .groupby(["session"]) \
                         .progress_apply(lambda x: suggest_clicks(x, covisit_clicks, top_clicks, model, index, aid2idx)) 

    pred_df_carts = test.sort_values(["session", "ts"]) \
                       .groupby(["session"]) \
                       .progress_apply(lambda x: suggest_carts(x, covisits_buy2buy, covisit_carts_orders, top_orders,
                                                               model, index, aid2idx))
    pred_df_orders = test.sort_values(["session", "ts"]) \
                         .groupby(["session"]) \
                         .progress_apply(lambda x: suggest_orders(x, covisits_buy2buy, covisit_carts_orders, top_orders,
                                                                model, index, aid2idx))
        

    clicks_pred_df = pd.DataFrame(pred_df_clicks.add_suffix("_clicks"), columns=["labels"]).reset_index()
    orders_pred_df = pd.DataFrame(pred_df_orders.add_suffix("_orders"), columns=["labels"]).reset_index()
    carts_pred_df = pd.DataFrame(pred_df_carts.add_suffix("_carts"), columns=["labels"]).reset_index()

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
    generate_combined_covisitation(data, config.chunk_size)
    generate_combined_covisitation(data, config.chunk_size, type='carts-orders')
    generate_combined_covisitation(data, time_diff=14*24*60*60, chunk_size=config.chunk_size, type='buy2buy')
    top_clicks, top_orders = get_top_clicks_orders(test)
    covisit_clicks = load_combined_covisitation(type='clicks')
    covisit_carts_orders = load_combined_covisitation(type='carts-orders')
    covisits_buy2buy = load_combined_covisitation(type='buy2buy')
    word2vec = load_model()
    index, aid2idx = build_index(word2vec)
    pred = generate_candidates(test, covisit_clicks, covisit_carts_orders, 
                               covisits_buy2buy, top_clicks, top_orders, word2vec, index, aid2idx)
    compute_validation_score(pred)
    if not config.local_validation:
        submit_file(message='separate covists with word2vec',
                    path=config.submission_path)

if __name__ == '__main__':
    main()