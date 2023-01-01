from collections import Counter, defaultdict
from typing import Tuple

import hydra
import numpy as np
import polars as pl
from gensim.models import Word2Vec
from gensim.test.utils import common_texts
from omegaconf import DictConfig, OmegaConf


def create_sentences(
    train: pl.DataFrame, test: pl.DataFrame
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    print("Creating sentences...")
    sentences_df = (
        pl.concat([train, test]).groupby("session").agg(pl.col("aid").alias("sentence"))
    )
    sentences = sentences_df["sentence"].to_list()
    return sentences[:100] if config.debug else sentences


def train_word2vec_model(sentences: pl.DataFrame) -> None:
    print("Training word2vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=config.vector_size,
        window=config.window,
        negative=config.negative,
        workers=config.workers,
        epochs=config.epochs,
        min_count=config.min_count,
    )
    model.save(f"{config.model_path}word2vec_windowl_{config.window}.model")
    print(
        f"Word2vec model saved to: {config.model_path}word2vec_windowl_{config.window}.model"
    )


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
    sentences = create_sentences(train, test)
    train_word2vec_model(sentences)


if __name__ == "__main__":
    main()
