import ast
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Tuple


class DataLoader:
    def __init__(self, config: dict, debug: bool) -> None:
        self.config = config
        self.debug = debug
        self.input_dir = Path(config["file_path"]["input_dir"])

    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        browsing_train = self.load_train_data("browsing_train")
        search_train = self.load_train_data("search_train")
        sku_to_content = self.load_train_data("sku_to_content")
        test = self.load_test_data(self.config["task"])

        if self.debug:
            train_session_ids = list(browsing_train["session_id_hash"].unique())
            train_session_ids = train_session_ids[:int(len(train_session_ids) * 0.01)]
            browsing_train = browsing_train[browsing_train["session_id_hash"].isin(train_session_ids)]
            search_train = search_train[search_train["session_id_hash"].isin(train_session_ids)]

        train = self._concat_browsing_and_search(browsing_train, search_train)
        return train, test, sku_to_content

    def load_train_data(self, data_type: str) -> pd.DataFrame:
        raw_file_path = self.input_dir / self.config["raw_file"][data_type]
        pkl_file_path = self.input_dir / self.config["pkl_file"][data_type]
        if pkl_file_path.exists():
            dataset = pd.read_pickle(pkl_file_path)
        else:
            dataset = pd.read_csv(raw_file_path)
            if data_type == "browsing_train":
                self._preprocessing_browsing_train(dataset)
            elif data_type == "search_train":
                self._preprocessing_search_train(dataset)
            elif data_type == "sku_to_content":
                self._preprocessing_sku_to_content(dataset)
            dataset.to_pickle(pkl_file_path)
        return dataset

    @staticmethod
    def _concat_browsing_and_search(browsing_df: pd.DataFrame, search_df: pd.DataFrame) -> pd.DataFrame:
        browsing_df["is_search"] = False
        search_df["is_search"] = True
        res = pd.concat([browsing_df, search_df], axis=0)
        res.sort_values(["session_id_hash", "server_timestamp_epoch_ms"], ascending=True, inplace=True)
        res.reset_index(drop=True, inplace=True)
        return res

    @staticmethod
    def _preprocessing_browsing_train(df: pd.DataFrame) -> None:
        df["server_timestamp"] = pd.to_datetime(
            df["server_timestamp_epoch_ms"], unit="ms",
        )
        df.fillna(np.nan, inplace=True)
        df.sort_values(["session_id_hash", "server_timestamp_epoch_ms"], ascending=True, inplace=True)

    @staticmethod
    def _preprocessing_search_train(df: pd.DataFrame) -> None:
        df["server_timestamp"] = pd.to_datetime(
            df["server_timestamp_epoch_ms"], unit="ms",
        )
        df["query_vector"] = (
            df["query_vector"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )
        df["product_skus_hash"] = (
            df["product_skus_hash"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )
        df["clicked_skus_hash"] = (
            df["clicked_skus_hash"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )
        df.fillna(np.nan, inplace=True)
        df.sort_values(["session_id_hash", "server_timestamp_epoch_ms"], ascending=True, inplace=True)

    @staticmethod
    def _preprocessing_sku_to_content(df: pd.DataFrame) -> None:
        df["image_vector"] = (
            df["image_vector"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )
        df["description_vector"] = (
            df["description_vector"]
            .apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else np.nan)
        )
        df["category_hash_first_level"] = (
            df["category_hash"]
            .apply(lambda x: x.split("/")[0] if isinstance(x, str) else np.nan)
        )
        df["category_hash_second_level"] = (
            df["category_hash"]
            .apply(lambda x: x.split("/")[1] if isinstance(x, str) and len(x.split("/")) >= 2 else np.nan)
        )
        df["category_hash_third_level"] = (
            df["category_hash"]
            .apply(lambda x: x.split("/")[2] if isinstance(x, str) and len(x.split("/")) >= 3 else np.nan)
        )
        df.fillna(np.nan, inplace=True)

    def load_test_data(self, task_type: str) -> pd.DataFrame:
        raw_file_path = self.input_dir / self.config["raw_file"]["test"]
        pkl_file_path = self.input_dir / self.config["pkl_file"]["test"]
        if pkl_file_path.exists():
            dataset = pd.read_pickle(pkl_file_path)
        else:
            with raw_file_path.open() as f:
                test = json.load(f)
            dataset = self._convert_json_to_dataframe(test)
            dataset.to_pickle(pkl_file_path)
        return dataset

    @staticmethod
    def _convert_json_to_dataframe(json_data: dict) -> pd.DataFrame:
        events = []
        for query_label in tqdm(json_data):
            query = query_label["query"]
            for event in query:
                events.append(event)
        res = pd.DataFrame(events)
        res["server_timestamp"] = pd.to_datetime(
            res["server_timestamp_epoch_ms"], unit="ms",
        )
        res.fillna(np.nan, inplace=True)
        res.sort_values(["session_id_hash", "server_timestamp_epoch_ms"], ascending=True, inplace=True)
        res.reset_index(drop=True, inplace=True)
        return res
