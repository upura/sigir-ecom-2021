import ast
import json
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def _concat_browsing_and_search(browsing_df: pd.DataFrame, search_df: pd.DataFrame) -> pd.DataFrame:
    browsing_df["is_search"] = False
    search_df["is_search"] = True
    res = pd.concat([browsing_df, search_df], axis=0)
    res.sort_values(["session_id_hash", "server_timestamp_epoch_ms"], ascending=True, inplace=True)
    res.reset_index(drop=True, inplace=True)
    return res


def _preprocessing_browsing_train(df: pd.DataFrame) -> pd.DataFrame:
    df["server_timestamp"] = pd.to_datetime(
        df["server_timestamp_epoch_ms"], unit="ms",
    )
    df.fillna(np.nan, inplace=True)
    df.sort_values(["session_id_hash", "server_timestamp_epoch_ms"], ascending=True, inplace=True)
    return df.reset_index(drop=True)


def _preprocessing_search_train(df: pd.DataFrame) -> pd.DataFrame:
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
    return df.reset_index(drop=True)


def _preprocessing_sku_to_content(df: pd.DataFrame) -> pd.DataFrame:
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
    return df.reset_index(drop=True)


def load_test_data() -> pd.DataFrame:
    with open('../session_rec_sigir_data/test/rec_test_phase_1.json', 'r') as f:
        test = json.load(f)
    dataset = _convert_json_to_dataframe(test)
    return dataset


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


def load_datasets() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    browsing_train = pd.read_csv('../session_rec_sigir_data/train/browsing_train.csv')
    search_train = pd.read_csv('../session_rec_sigir_data/train/search_train.csv')
    sku_to_content = pd.read_csv('../session_rec_sigir_data/train/sku_to_content.csv')

    browsing_train = _preprocessing_browsing_train(browsing_train)
    search_train = _preprocessing_search_train(search_train)
    sku_to_content = _preprocessing_sku_to_content(sku_to_content)
    test = load_test_data()

    train = _concat_browsing_and_search(browsing_train, search_train)
    return train, test, sku_to_content


class MacOSFile(object):
    # https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def read(self, n):
        # print("reading total_bytes=%s" % n, flush=True)
        if n >= (1 << 31):
            buffer = bytearray(n)
            idx = 0
            while idx < n:
                batch_size = min(n - idx, 1 << 31 - 1)
                # print("reading bytes [%s,%s)..." % (idx, idx + batch_size), end="", flush=True)
                buffer[idx:idx + batch_size] = self.f.read(batch_size)
                # print("done.", flush=True)
                idx += batch_size
            return buffer
        return self.f.read(n)

    def write(self, buffer):
        n = len(buffer)
        print("writing total_bytes=%s..." % n, flush=True)
        idx = 0
        while idx < n:
            batch_size = min(n - idx, 1 << 31 - 1)
            print("writing bytes [%s, %s)... " % (idx, idx + batch_size), end="", flush=True)
            self.f.write(buffer[idx:idx + batch_size])
            print("done.", flush=True)
            idx += batch_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, MacOSFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(MacOSFile(f))
