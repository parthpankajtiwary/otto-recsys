import ast
import gc
import os
import pickle
import warnings
from typing import List, Tuple

import cudf
import numpy as np
import pandas as pd
import polars as pl
from tqdm import tqdm

warnings.filterwarnings("ignore")

import hydra
from omegaconf import DictConfig, OmegaConf


def load_data() -> pd.DataFrame:
    """Load data from parquet files."""
    if config.local_validation:
        train = cudf.read_parquet(config.validation_path + config.train_file)
        test = cudf.read_parquet(config.validation_path + config.test_file)
    else:
        train = cudf.read_parquet(config.data_path + config.train_file)
        test = cudf.read_parquet(config.data_path + config.test_file)
    data = cudf.concat([train, test])
    if config.debug:
        data = data.sample(frac=config.fraction, random_state=config.random_state)
    return data, test


def process_covisitation_in_chunks(
    data: pd.DataFrame, chunk_size: int, type: str = "clicks"
) -> pd.DataFrame:
    """Process data in chunks and return a dataframe."""
    tmp = []
    data = data.set_index("session")
    sessions = data.index.unique()
    _min, _max = data.ts.min(), data.ts.max()

    for i in tqdm(range(0, sessions.shape[0], chunk_size)):
        df = data.loc[
            sessions[i] : sessions[min(sessions.shape[0] - 1, i + chunk_size - 1)]
        ].reset_index()
        if type == "buy2buy":
            df = df.loc[(df["type"] == 1) | (df["type"] == 2)]
        df = df.sort_values(["session", "ts"], ascending=[True, False])
        df = df.reset_index(drop=True)
        df["n"] = df.groupby("session").cumcount()
        df = df.loc[df.n < config.n_samples].drop("n", axis=1)
        df = df.merge(df, on="session")

        if type == "buy2buy":
            df = df.loc[
                ((df.ts_x - df.ts_y).abs() < eval(config.time_diff))
                & (df.aid_x != df.aid_y)
            ]
            df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(
                ["session", "aid_x", "aid_y"]
            )
            df["wgt"] = 1

        elif type == "carts-orders":
            df = df.loc[
                ((df.ts_x - df.ts_y).abs() < eval(config.time_diff) // 7)
                & (df.aid_x != df.aid_y)
            ]
            df = df[["session", "aid_x", "aid_y", "type_y"]].drop_duplicates(
                ["session", "aid_x", "aid_y"]
            )
            df["wgt"] = df.type_y.map(eval(str(config.type_weight)))
        elif type == "clicks":
            df = df.loc[
                ((df.ts_x - df.ts_y).abs() < eval(config.time_diff) // 7)
                & (df.aid_x != df.aid_y)
            ]
            df = df[["session", "aid_x", "aid_y", "ts_x"]].drop_duplicates(
                ["session", "aid_x", "aid_y"]
            )
            df["wgt"] = 1 + 3 * ((df.ts_x - _min) / (_max - _min))
        df = df[["aid_x", "aid_y", "wgt"]]
        df.wgt = df.wgt.astype("float32")
        df = df.groupby(["aid_x", "aid_y"]).wgt.sum()

        tmp.append(df.reset_index())

        del df
        gc.collect()

    return tmp


def combine_covisitation_chunks(tmp: List[Tuple[str, str, int]]):
    """Combine covisitation chunks and return a dataframe."""
    tmp = list(map(lambda x: pl.DataFrame(x.to_pandas()), tmp))
    tmp = pl.concat(tmp)
    return tmp


def generate_combined_covisitation(
    data: pd.DataFrame, chunk_size: int, type="clicks"
) -> None:
    """Generate combined covisitation."""
    print("Processing co-visitation matrix in chunks...")
    tmp = process_covisitation_in_chunks(data, chunk_size, type)
    tmp = combine_covisitation_chunks(tmp)
    print("Generating combined covisitation matrix...")
    tmp = (
        (tmp.groupby(["aid_x", "aid_y"]).agg(pl.col("wgt").sum()))
        .to_pandas()
        .reset_index()
    )
    tmp = tmp.sort_values(["aid_x", "wgt"], ascending=[True, False])
    tmp = tmp.reset_index(drop=True)
    tmp["n"] = tmp.groupby("aid_x").aid_y.cumcount()
    if type == "clicks":
        tmp = tmp.loc[tmp.n < config.n_top_clicks].drop("n", axis=1)
    else:
        tmp = tmp.loc[tmp.n < config.n_top].drop("n", axis=1)
    df = tmp.groupby("aid_x").aid_y.apply(list)
    save_combined_covisitation(df, type)
    print("Combined covisitation matrix saved.")
    del tmp, df


def save_combined_covisitation(df: pd.DataFrame, type="clicks") -> None:
    """Save combined covisitation."""
    with open(config.data_path + f"top_20_{type}_v{config.version}.pkl", "wb") as f:
        pickle.dump(df.to_dict(), f)


"""Main module."""


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global config
    config = cfg
    data, _ = load_data()
    generate_combined_covisitation(data, config.chunk_size)
    generate_combined_covisitation(data, config.chunk_size, type="carts-orders")
    generate_combined_covisitation(data, chunk_size=config.chunk_size, type="buy2buy")


if __name__ == "__main__":
    main()
