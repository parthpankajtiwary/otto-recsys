import itertools
from sys import displayhook
import time
import warnings
from collections import Counter

import numpy as np
import polars as pl

warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf


def load_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    candidate_df_clicks = pl.read_parquet(
        f"{config.artifact_path}candidate_df_clicks.parquet"
    )
    item_features = pl.read_parquet(
        f"{config.artifact_path}item_features.parquet"
    )
    user_features = pl.read_parquet(
        f"{config.artifact_path}user_features.parquet"
    )
    return candidate_df_clicks, item_features, user_features


def join(candidate_df: pl.DataFrame, features: list, on: str) -> pl.DataFrame:
    return candidate_df.join(features, on=on, how="left").fill_null(-1)


def cast_int32(df: pl.DataFrame, columns: list) -> pl.DataFrame:
    for column in columns:
        df = df.with_column(pl.col(column).cast(pl.Int32))
    return df


"""Main module."""


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global config, run
    config = cfg
    candidate_df_clicks, item_features, user_features = load_data()
    print(candidate_df_clicks.groupby("session").count().head())
    assert item_features["aid"].n_unique() > 1800000
    assert user_features["session"].n_unique() > 1800000
    candidate_df_clicks = cast_int32(candidate_df_clicks, ["aid", "session"])
    print("Candidate df clicks shape: ", candidate_df_clicks.shape)
    candidate_df_clicks = join(candidate_df_clicks, item_features, on="aid")
    print("Candidate df clicks with item features shape: ", candidate_df_clicks.shape)
    candidate_df_clicks_with_features = join(
        candidate_df_clicks, user_features, on="session"
    )
    print("Candidate df clicks with user features shape: ", candidate_df_clicks_with_features.shape)
    candidate_df_clicks_with_features.write_parquet(
        f"{config.artifact_path}candidate_df_clicks_with_features.parquet"
    )
    print(candidate_df_clicks_with_features.head())
    assert candidate_df_clicks.shape[0] == candidate_df_clicks_with_features.shape[0]
    print("Saved candidate df clicks with features to: ", f"{config.artifact_path}candidate_df_clicks_with_features.parquet")


if __name__ == "__main__":
    main()
