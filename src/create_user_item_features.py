from typing import Tuple

import hydra
import numpy as np
import pandas as pd
import polars as pl
from omegaconf import DictConfig, OmegaConf


def create_user_features(train: pl.DataFrame, test: pl.DataFrame) -> None:
    data = test
    data = convert_seconds_to_milliseconds(data)
    data = create_user_recency_features(data)
    data = data.fill_null(-1)
    print(data.head())
    print("User features shape: ", data.shape)
    data.write_parquet(config.artifact_path + "user_features.parquet")


def create_item_features(train: pl.DataFrame, test: pl.DataFrame) -> None:
    data = pl.concat([train, test])
    data = convert_seconds_to_milliseconds(data)
    df_recency_features = create_item_recency_features(data)
    df_time_features = create_time_features(data)
    data = df_recency_features.join(df_time_features, on="aid", how="left").sort("aid")
    data = data.fill_null(-1)
    print(data.head())
    print("Item features shape: ", data.shape)
    data.write_parquet(config.artifact_path + "item_features.parquet")


def convert_seconds_to_milliseconds(df: pl.DataFrame) -> pl.DataFrame:
    MILLISECONDS_IN_SECOND = 1000

    df = df.with_columns(
        [
            (pl.col("ts").cast(pl.Int64) * MILLISECONDS_IN_SECOND)
            .cast(pl.Datetime)
            .dt.with_time_unit("ms")
            .alias("datetime")
        ]
    )
    return df


def create_item_recency_features(df: pl.DataFrame) -> pl.DataFrame:
    one_day = 24 * 60 * 60
    seven_days = 7 * one_day
    df = (
        df.groupby("aid")
        .agg(
            [
                (pl.col("type") == 0).sum().alias("item_n_clicks"),
                (pl.col("type") == 1).sum().alias("item_n_carts"),
                (pl.col("type") == 2).sum().alias("item_n_orders"),
                ((pl.col("type") == 0) & (pl.col("ts") > pl.col("ts").max() - one_day))
                .sum()
                .alias("item_n_clicks_24h"),
                ((pl.col("type") == 1) & (pl.col("ts") > pl.col("ts").max() - one_day))
                .sum()
                .alias("item_n_carts_24h"),
                ((pl.col("type") == 2) & (pl.col("ts") > pl.col("ts").max() - one_day))
                .sum()
                .alias("item_n_orders_24h"),
                # number of clicks in last 7 days
                (
                    (pl.col("type") == 0)
                    & (pl.col("ts") > pl.col("ts").max() - seven_days)
                )
                .sum()
                .alias("item_n_clicks_7d"),
                # number of carts in last 7 days
                (
                    (pl.col("type") == 1)
                    & (pl.col("ts") > pl.col("ts").max() - seven_days)
                )
                .sum()
                .alias("item_n_carts_7d"),
                # number of orders in last 7 days
                (
                    (pl.col("type") == 2)
                    & (pl.col("ts") > pl.col("ts").max() - seven_days)
                )
                .sum()
                .alias("item_n_orders_7d"),
            ]
        )
        .fill_null(-1)
        .sort("aid")
    )
    return df


def create_time_features(df: pl.DataFrame) -> pl.DataFrame:
    df = (
        df.groupby("aid")
        .agg(
            [
                # good to calculate for last 24 hours and 7d
                # average time between clicks
                # average click hour
                pl.col("datetime")
                .filter(pl.col("type") == 0)
                .dt.hour()
                .mean()
                .alias("item_avg_click_hour"),
                # average cart hour
                pl.col("datetime")
                .filter(pl.col("type") == 1)
                .dt.hour()
                .mean()
                .alias("item_avg_cart_hour"),
                # average order hour
                pl.col("datetime")
                .filter(pl.col("type") == 2)
                .dt.hour()
                .mean()
                .alias("item_avg_order_hour"),
                # average click day of month
                pl.col("datetime")
                .filter(pl.col("type") == 0)
                .dt.day()
                .mean()
                .alias("item_avg_click_day_of_month"),
                # average cart day of month
                pl.col("datetime")
                .filter(pl.col("type") == 1)
                .dt.day()
                .mean()
                .alias("item_avg_cart_day_of_month"),
                # average order day of month
                pl.col("datetime")
                .filter(pl.col("type") == 2)
                .dt.day()
                .mean()
                .alias("item_avg_order_day_of_month"),
            ]
        )
        .fill_null(-1)
        .sort("aid")
    )
    return df


def create_user_recency_features(df: pl.DataFrame) -> pl.DataFrame:
    one_day = 24 * 60 * 60
    seven_days = 7 * one_day
    df = (
        df.groupby("session")
        .agg(
            [
                (pl.col("type") == 0).sum().alias("user_n_clicks"),
                (pl.col("type") == 1).sum().alias("user_n_carts"),
                (pl.col("type") == 2).sum().alias("user_n_orders"),
                ((pl.col("type") == 0) & (pl.col("ts") > pl.col("ts").max() - one_day))
                .sum()
                .alias("user_n_clicks_24h"),
                ((pl.col("type") == 1) & (pl.col("ts") > pl.col("ts").max() - one_day))
                .sum()
                .alias("user_n_carts_24h"),
                ((pl.col("type") == 2) & (pl.col("ts") > pl.col("ts").max() - one_day))
                .sum()
                .alias("user_n_orders_24h"),
                # number of clicks in last 7 days
                (
                    (pl.col("type") == 0)
                    & (pl.col("ts") > pl.col("ts").max() - seven_days)
                )
                .sum()
                .alias("user_n_clicks_7d"),
                # number of carts in last 7 days
                (
                    (pl.col("type") == 1)
                    & (pl.col("ts") > pl.col("ts").max() - seven_days)
                )
                .sum()
                .alias("user_n_carts_7d"),
                # number of orders in last 7 days
                (
                    (pl.col("type") == 2)
                    & (pl.col("ts") > pl.col("ts").max() - seven_days)
                )
                .sum()
                .alias("user_n_orders_7d"),
                # number of unique items in last 7 days
                (pl.col("aid").filter(pl.col("ts") > pl.col("ts").max() - seven_days))
                .n_unique()
                .alias("user_n_unique_items_7d"),
                # number of unique items in last 24 hours
                (pl.col("aid").filter(pl.col("ts") > pl.col("ts").max() - one_day))
                .n_unique()
                .alias("user_n_unique_items_24h"),
                # session length
                pl.col("ts").n_unique().alias("user_session_length"),
                # average click hour
                pl.col("datetime")
                .filter(pl.col("type") == 0)
                .dt.hour()
                .mean()
                .alias("user_avg_click_hour"),
                # average cart hour
                pl.col("datetime")
                .filter(pl.col("type") == 1)
                .dt.hour()
                .mean()
                .alias("user_avg_cart_hour"),
                # average order hour
                pl.col("datetime")
                .filter(pl.col("type") == 2)
                .dt.hour()
                .mean()
                .alias("user_avg_order_hour"),
            ]
        )
        .fill_null(-1)
        .sort("session")
    )
    return df


def load_data() -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load data from parquet files."""
    if config.local_validation:
        train = pl.read_parquet(config.validation_path + config.train_file)
        test = pl.read_parquet(config.validation_path + config.test_file)
    else:
        train = pl.read_parquet(config.data_path + config.train_file)
        test = pl.read_parquet(config.data_path + config.test_file)
    return train, test


"""Main module."""


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    global config
    config = cfg
    train, test = load_data()
    create_user_features(train, test)
    create_item_features(train, test)


if __name__ == "__main__":
    main()
