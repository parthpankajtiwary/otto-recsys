import itertools
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from annoy import AnnoyIndex
from gensim.models import Word2Vec
from tqdm import tqdm

warnings.filterwarnings("ignore")
from pandarallel import pandarallel

pandarallel.initialize(
    progress_bar=True,
    nb_workers=10,
)
from typing import Dict, List, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load data from parquet files."""
    if config.local_validation:
        train = pd.read_parquet(config.validation_path + config.train_file)
        test = pd.read_parquet(config.validation_path + config.test_file)
    else:
        train = pd.read_parquet(config.data_path + config.train_file)
        test = pd.read_parquet(config.data_path + config.test_file)
    data = pd.concat([train, test])
    if config.debug:
        data = data.sample(frac=config.fraction, random_state=config.random_state)
    return data, test


def load_model():
    """Load word2vec model."""
    if not config.word2vec:
        return None
    print("Loading word2vec model...")
    model = Word2Vec.load(config.model_path + "word2vec.model")
    print("Model loaded from path: ", config.model_path + "word2vec.model")
    return model


def build_index(model, n_trees=100) -> Tuple[AnnoyIndex, Dict[str, int]]:
    """Build index for word2vec model."""
    if config.word2vec:
        print("Building index for word2vec model...")
        aid2idx = {aid: i for i, aid in enumerate(model.wv.index_to_key)}
        index = AnnoyIndex(model.wv.vector_size, metric="euclidean")
        for idx in aid2idx.values():
            index.add_item(idx, model.wv.vectors[idx])
        index.build(n_trees=n_trees)
        return index, aid2idx
    else:
        return None, None


def get_nns(
    model: Word2Vec,
    index: AnnoyIndex,
    product: str,
    aid2idx: Dict[str, int],
    n: int = 21,
) -> List[str]:
    """Get nearest neighbors for a given aid."""
    try:
        idx = aid2idx[product]
        nns = index.get_nns_by_item(idx, n)[1:]
        nns = [model.wv.index_to_key[idx] for idx in nns]
    except Exception:
        nns = []
    return nns


def get_top_clicks_orders(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    top_clicks = (
        df.loc[df["type"] == config.type_labels["clicks"], "aid"]
        .value_counts()
        .index.values[:20]
    )
    top_orders = (
        df.loc[df["type"] == config.type_labels["orders"], "aid"]
        .value_counts()
        .index.values[:20]
    )
    return top_clicks, top_orders


def load_combined_covisitation(type: str = "clicks") -> pd.DataFrame:
    top_20 = pd.read_pickle(config.data_path + f"top_20_{type}_v{config.version}.pkl")
    print(f"Size of top_20_{type}:", len(top_20))
    return top_20


def suggest_clicks(
    df: pd.DataFrame, top_20: List[str], top_clicks: List[str]
) -> List[str]:  # sourcery skip: extract-duplicate-method
    products = df.aid.tolist()
    types = df.type.tolist()
    unique_products = list(dict.fromkeys(products[::-1]))

    if len(unique_products) >= 20:
        weights = np.logspace(0.1, 1, len(products), base=2, endpoint=True) - 1
        products_tmp = Counter()

        for product, weight, _type in zip(products, weights, types):
            products_tmp[product] += weight * config.type_weight[_type]

        return [product for product, _ in products_tmp.most_common(50)]

    products_1 = list(
        itertools.chain(
            *[top_20[product] for product in unique_products if product in top_20]
        )
    )

    top_products = [
        product
        for product, _ in Counter(products_1).most_common(50)
        if product not in unique_products
    ]

    if config.word2vec:
        products_2 = list(
            itertools.chain(
                *[
                    get_nns(model, index, product, aid2idx)
                    for product in top_products[: config.n_recent]
                    + unique_products[: config.n_recent]
                ]
            )
        )
        top_word2vec = [
            product
            for product, _ in Counter(products_2).most_common(50)
            if product not in unique_products
        ]
        return list(set(unique_products + top_products + top_word2vec))
    else:
        result = unique_products + top_products
        return result + list(top_clicks)


def generate_candidates(
    test: pd.DataFrame,
    covisit_clicks: pd.DataFrame,
    covisit_carts_orders: pd.DataFrame,
    covisits_buy2buy: pd.DataFrame,
    top_clicks: pd.DataFrame,
    top_orders: pd.DataFrame,
) -> None:

    pred_df_clicks = (
        test.sort_values(["session", "ts"])
        .groupby(["session"])
        .parallel_apply(lambda x: suggest_clicks(x, covisit_clicks, top_clicks))
    )
    # pred_df_orders = (
    #     test.sort_values(["session", "ts"])
    #     .groupby(["session"])
    #     .parallel_apply(
    #         lambda x: suggest_orders(
    #             x,
    #             covisits_buy2buy,
    #             covisit_carts_orders,
    #             top_orders,
    #         )
    #     )
    # )
    candidate_df_clicks = pred_df_clicks.explode().reset_index()
    candidate_df_clicks.columns = ["session", "aid"]

    print("Candidate df clicks shape: ", candidate_df_clicks.shape)
    print(
        "Candidate df clicks unique sessions: ", candidate_df_clicks.session.nunique()
    )
    print(
        "Min max session: ",
        candidate_df_clicks.session.min(),
        candidate_df_clicks.session.max(),
    )

    candidate_df_clicks.to_parquet(config.artifact_path + "candidate_df_clicks.parquet")


"""Main module."""


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global config, run
    config = cfg
    data, test = load_data()
    top_clicks, top_orders = get_top_clicks_orders(test)
    covisit_clicks = load_combined_covisitation(type="clicks")
    covisit_carts_orders = load_combined_covisitation(type="carts-orders")
    covisits_buy2buy = load_combined_covisitation(type="buy2buy")
    # idiotic hack to make word2vec work with parallel_apply
    global index, aid2idx, model
    model = load_model()
    index, aid2idx = build_index(model)

    generate_candidates(
        test,
        covisit_clicks,
        covisit_carts_orders,
        covisits_buy2buy,
        top_clicks,
        top_orders,
    )


if __name__ == "__main__":
    main()
