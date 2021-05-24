import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, List
from sklearn.preprocessing import normalize
from tqdm import tqdm
from annoy import AnnoyIndex


def normalize_l2(x):
    x_l2_norm = np.linalg.norm(x, ord=2)
    return x / x_l2_norm


class Preprocessor:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.interim_dir = Path(config["file_path"]["interim_dir"])
        self.encode_cols = [
            "session_id_hash",
            "product_sku_hash",
            "product_action",
            "hashed_url",
            "price_bucket",
            "number_of_category_hash",
            "category_hash_first_level",
            "category_hash_second_level",
            "category_hash_third_level",
            "event_type",
        ]
        self.label_to_index_dict = {}
        self.index_to_label_dict = {}

    def run(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        sku_to_content: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train["is_test"] = False
        test["is_test"] = True
        total = pd.concat([train, test], axis=0)
        total.loc[total["is_search"] == 1, "event_type"] = "search"

        self.get_query_features(total)
        total = self._filter_out(total)

        self.preprocessing_sku_to_content(sku_to_content)
        total = self._replace_items(total, sku_to_content)

        total = pd.merge(total, sku_to_content, on=["product_sku_hash"], how="left")

        self.get_time_features(total)
        self._label_encoding(total)
        self.fillna(total)

        train_preprocessed = total[total["is_test"] == False]
        test_preprocessed = total[total["is_test"] == True]
        return train_preprocessed, test_preprocessed

    @staticmethod
    def preprocessing_sku_to_content(sku_to_content: pd.DataFrame) -> None:
        sku_to_content["number_of_category_hash"] = (
            sku_to_content["category_hash"]
            .apply(
                lambda x: len(x.split("/")) if isinstance(x, str) else np.nan
            )
        )
        sku_to_content["description_vector_normalized"] = (
            sku_to_content["description_vector"].apply(
                lambda x: list(normalize_l2(x)) 
                if isinstance(x, list) else np.nan)
        )
        sku_to_content["image_vector_normalized"] = (
            sku_to_content["image_vector"].apply(
                lambda x: list(normalize_l2(x)) 
                if isinstance(x, list) else np.nan)
        )
        sku_to_content["item_vector"] = (
            sku_to_content.apply(
                lambda x:
                normalize_l2(x["image_vector_normalized"] + x["description_vector_normalized"])
                if isinstance(x["image_vector_normalized"], list) and isinstance(x["description_vector_normalized"], list)
                else np.nan, axis=1)
        )

    @staticmethod
    def fillna(df: pd.DataFrame) -> None:
        df["description_vector"] = (
            df["description_vector"]
            .apply(lambda x: x if isinstance(x, list) else [0.5] * 50)
        )
        df["image_vector"] = (
            df["image_vector"]
            .apply(lambda x: x if isinstance(x, list) else [0.5] * 50)
        )

    def get_time_features(self, df: pd.DataFrame) -> None:
        df["elapsed_time"] = (
            (
                df["server_timestamp_epoch_ms"]
                - df.groupby("session_id_hash")["server_timestamp_epoch_ms"].shift()
            ).fillna(0) / 1000   # ms -> sec
        )
        df["elapsed_time"] = (
            (df["elapsed_time"].astype(int) + 1)
            .clip(lower=1, upper=self.config["encoder_params"]["size_elapsed_time"] - 1)
        )

        df["hour"] = (df["server_timestamp"].dt.hour + 1).astype("int8")
        df["weekday"] = (df["server_timestamp"].dt.weekday + 1).astype("int8")
        df["weekend"] = df["weekday"].isin([6, 7]).astype("int8") + 1

    @staticmethod
    def get_query_features(df: pd.DataFrame) -> None:
        pass

    def _label_encoding(self, df: pd.DataFrame) -> None:
        for col in self.encode_cols:
            index_series, label_to_index, index_to_label = self._label_encode_series(df[col].astype(str))
            df[col] = index_series
            self.label_to_index_dict[col] = label_to_index
            self.index_to_label_dict[col] = index_to_label

    @staticmethod
    def _label_encode_series(series: pd.Series) -> Tuple[pd.Series, dict, dict]:
        """https://github.com/coveooss/SIGIR-ecom-data-challenge/blob/main/baselines/create_session_rec_input.py#L31-L42
        """
        labels = sorted(set(series.dropna().unique()), reverse=True)   # avoid null value
        label_to_index = {l: idx + 1 for idx, l in enumerate(labels)}  # 0: padding id
        index_to_label = {v: k for k, v in label_to_index.items()}
        return series.map(label_to_index), label_to_index, index_to_label

    def _filter_out(self, df: pd.DataFrame) -> pd.DataFrame:
        # rows with pageview generated by item interaction
        original_rows = len(df)
        item_interactions = (
            df[df["product_action"].notnull()]
            [["session_id_hash", "hashed_url", "server_timestamp_epoch_ms"]]
            .drop_duplicates()
        )
        item_interactions["is_interaction"] = 1
        df = pd.merge(
            df,
            item_interactions[["session_id_hash", "hashed_url", "server_timestamp_epoch_ms", "is_interaction"]],
            on=["session_id_hash", "hashed_url", "server_timestamp_epoch_ms"],
            how="left",
        )
        assert original_rows == len(df), "original_rows != len(df)"
        df = df[~((df["is_interaction"] == 1) & (df["event_type"] == "pageview"))]

        return df

    @staticmethod
    def _replace_items(df: pd.DataFrame, sku_to_content: pd.DataFrame) -> pd.DataFrame:
        train_item_set = set(df.query("is_test == False")["product_sku_hash"].unique())
        test_item_set = set(df.query("is_test == True")["product_sku_hash"].unique())
        test_only_item_set = test_item_set - train_item_set
        target_item_set = test_only_item_set
        candidacies_item_set = train_item_set

        vector_df = sku_to_content[["product_sku_hash", "item_vector"]].dropna().reset_index(drop=True)
        candidacies_vector_df = vector_df[vector_df["product_sku_hash"].isin(candidacies_item_set)]
        target_item_vector_df = vector_df[vector_df["product_sku_hash"].isin(target_item_set)]
        index_to_item = {idx: l for idx, l in enumerate(candidacies_vector_df["product_sku_hash"])}
        t = AnnoyIndex(100, "angular")
        for i, v in enumerate(candidacies_vector_df["item_vector"].values):
            t.add_item(i, v)
        t.build(10)

        mapping = {}
        for i, row in target_item_vector_df.iterrows():
            item = row["product_sku_hash"]
            item_vector = row["item_vector"]
            candidacate_index = t.get_nns_by_vector(item_vector, 1, include_distances=False)[0]
            most_sim_item = index_to_item[candidacate_index]
            mapping[item] = most_sim_item
        df["product_sku_hash"] = df["product_sku_hash"].replace(mapping)

        target_item_set_no_vector = target_item_set - set(target_item_vector_df["product_sku_hash"])
        df.loc[df["product_sku_hash"].isin(target_item_set_no_vector), "product_sku_hash"] = np.nan

        return df

    @staticmethod
    def get_session_sequences(df: pd.DataFrame) -> Dict[int, Dict[str, List]]:
        session_seqs = (
            df
            .groupby("session_id_hash")
            .agg({
                "product_sku_hash": list,
                "elapsed_time": list,
                "product_action": list,
                "hashed_url": list,
                "price_bucket": list,
                "number_of_category_hash": list,
                "category_hash_first_level": list,
                "category_hash_second_level": list,
                "category_hash_third_level": list,
                "description_vector": list,
                "image_vector": list,
                "event_type": list,
                "hour": list,
                "weekday": list,
                "weekend": list,
            })
            .to_dict(orient="index") 
        )
        return session_seqs
