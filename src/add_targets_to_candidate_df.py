import itertools
import time
import warnings
from collections import Counter

import numpy as np
import polars as pl
import pandas as pd

warnings.filterwarnings("ignore")

from typing import Dict, List, Tuple
import hydra
from omegaconf import DictConfig, OmegaConf


def load_data() -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    candidate_df_clicks_with_features = pl.read_parquet(
        f"{config.artifact_path}candidate_df_clicks_with_features.parquet"
    )
    target_labels = pl.read_parquet(
        f"{config.validation_path}test_labels.parquet"
    )
    return candidate_df_clicks_with_features, target_labels


def process_targets(df: pd.DataFrame, type: str) -> pd.DataFrame:
    print(df.head())
    targets = df.loc[ df['type']==type ]
    aids = targets.ground_truth.explode().astype('int32').rename('aid')
    targets = targets[['session']].astype('int32')
    targets = targets.merge(aids, left_index=True, right_index=True, how='left')
    targets[type] = 1
    print(f"Targets {type} shape: ", targets.shape[0])
    return pl.DataFrame(targets)


def join(candidate_df: pl.DataFrame, features: list, on: str) -> pl.DataFrame:
    return candidate_df.join(features, on=on, how="left").fill_null(0)

"""Main module."""


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global config, run
    config = cfg
    candidate_df_clicks_with_features, target_labels = load_data()
    print("Candidate df clicks with features shape: ", candidate_df_clicks_with_features.shape)
    targets_df = process_targets(target_labels.to_pandas(), 'clicks')
    print(targets_df.head())
    candidate_df_clicks_with_features_with_targets = join(
        candidate_df_clicks_with_features, targets_df, on=["session", "aid"]
    )
    print("Candidate df clicks with features with targets shape: ", candidate_df_clicks_with_features_with_targets.shape)
    # print number of rows with target 1
    print("Number of rows with target 1: ", candidate_df_clicks_with_features_with_targets['clicks'].sum())
    print(candidate_df_clicks_with_features_with_targets.head())
    candidate_df_clicks_with_features_with_targets.write_parquet(
        f"{config.artifact_path}candidate_df_clicks_with_features_with_targets.parquet"
    )


if __name__ == "__main__":
    main()
