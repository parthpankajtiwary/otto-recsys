import pandas as pd


class config:
    data_path = "data/"
    validation_path = "data/local_validation/"


def main():
    candidate_df = pd.read_parquet(
        config.validation_path + "candidate_df_with_user_item_features.parquet"
    )
    test_labels = pd.read_parquet(config.validation_path + "test_labels.parquet")
    test_labels = test_labels.loc[test_labels["type"] == "clicks"]
    items = test_labels.ground_truth.explode().astype("int32").rename("aid")
    test_labels = test_labels[["session"]].astype("int32")
    test_labels = test_labels.merge(
        items, left_index=True, right_index=True, how="left"
    )
    test_labels["click"] = 1
    print("Merge candidate_df and test_labels")
    candidate_df = candidate_df.merge(
        test_labels, on=["session", "aid"], how="left"
    ).fillna(0)
    candidate_df.to_parquet(
        config.validation_path
        + "candidate_df_with_user_item_features_and_target.parquet"
    )


if __name__ == "__main__":
    main()
