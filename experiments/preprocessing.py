import pandas as pd
from sklearn.model_selection import StratifiedKFold

from datasets import pickle_load
from preprocessors import Preprocessor


if __name__ == '__main__':
    train = pickle_load('../session_rec_sigir_data/prepared/train.pkl')
    test = pickle_load('../session_rec_sigir_data/prepared/test.pkl')
    sku_to_content = pickle_load('../session_rec_sigir_data/prepared/sku_to_content.pkl')
    test_session_ids = set(test["session_id_hash"].unique())

    pr = Preprocessor()
    train_preprocessed, test_preprocessed = pr.run(train, test, sku_to_content)

    train_session_info = train_preprocessed[
        ["session_id_hash", "session_len_count"]
    ].drop_duplicates()
    cv = StratifiedKFold(shuffle=True,
                         n_splits=5,
                         random_state=71)
    folds = cv.split(
        train_session_info,
        pd.cut(
            train_session_info["session_len_count"],
            5,
            labels=False,
        ),
    )

    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        if i_fold in (0, ):
            train_session_ids = train_session_info.iloc[trn_idx]["session_id_hash"].tolist()
            val_session_ids = train_session_info.iloc[val_idx]["session_id_hash"].tolist()
            train_session_seqs = pr.get_session_sequences(
                train_preprocessed[train_preprocessed["session_id_hash"].isin(train_session_ids)]
            )
            val_session_seqs = pr.get_session_sequences(
                train_preprocessed[train_preprocessed["session_id_hash"].isin(val_session_ids)]
            )
            # Create features here
