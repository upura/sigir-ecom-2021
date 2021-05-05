from typing import Tuple, Dict, List

import pandas as pd


class Preprocessor:
    def __init__(self) -> None:
        self.encode_cols = [
            "session_id_hash",
            "product_sku_hash",
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

        total = self._filter_out(total)
        self._label_encoding(total)
        total["elapsed_time"] = (
            (
                total["server_timestamp_epoch_ms"]
                - total.groupby("session_id_hash")["server_timestamp_epoch_ms"].shift()
            ).fillna(0) / 1000   # ms -> sec
        )
        train_preprocessed = total[total["is_test"] == False]
        test_preprocessed = total[total["is_test"] == True]
        return train_preprocessed, test_preprocessed

    def _label_encoding(self, df: pd.DataFrame) -> None:
        for col in self.encode_cols:
            index_series, label_to_index, index_to_label = self._label_encode_series(df[col])
            df[col] = index_series
            self.label_to_index_dict[col] = label_to_index
            self.index_to_label_dict[col] = index_to_label

    @staticmethod
    def _label_encode_series(series: pd.Series) -> Tuple[pd.Series, dict, dict]:
        """https://github.com/coveooss/SIGIR-ecom-data-challenge/blob/main/baselines/create_session_rec_input.py#L31-L42
        """
        labels = set(series.dropna().unique())   # avoid null value
        label_to_index = {l: idx + 1 for idx, l in enumerate(labels)}  # 0: padding id
        index_to_label = {v: k for k, v in label_to_index.items()}
        return series.map(label_to_index), label_to_index, index_to_label

    def _filter_out(self, df: pd.DataFrame) -> pd.DataFrame:
        # `remove from cart` events to avoid feeding them to session_rec as positive signals
        df = df[df['product_action'] != 'remove']
        # rows with null product_sku_hash
        df = df.dropna(subset=['product_sku_hash'])
        # sessions with only one action (train only)
        df["session_len_count"] = df.groupby("session_id_hash")["session_id_hash"].transform("count")
        df = df.loc[~((df["session_len_count"] < 2) & (df["is_test"] == False))]
        # unseen from train data
        train_item_index_set = set(df.query("is_test == False")["product_sku_hash"].unique())
        df = df[df["product_sku_hash"].isin(train_item_index_set)]
        return df

    @staticmethod
    def get_session_sequences(df: pd.DataFrame) -> Dict[int, Dict[str, List]]:
        session_seqs = (
            df
            .groupby("session_id_hash")
            .agg({
                "product_sku_hash": list,
                "elapsed_time": list,
            })
            .to_dict(orient="index") 
        )
        return session_seqs
