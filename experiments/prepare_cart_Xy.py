import sys

import numpy as np
import pandas as pd

sys.path.append('../')
from ayniy.utils import Data
from datasets import pickle_load


if __name__ == '__main__':
    for nb in [0, 2, 4, 6, 8, 10]:
        print('****** Starting nb==', nb)
        df_pos = pd.read_pickle(f'../session_rec_sigir_data/prepared/train_pos_nb{nb}.pkl')
        df_neg = pd.read_pickle(f'../session_rec_sigir_data/prepared/train_neg_nb{nb}.pkl')

        df_pos['label'] = 1
        df_neg['label'] = 0

        df_train = pd.concat([df_pos, df_neg], axis=0)
        X_train = df_train.groupby('session_id_hash').agg({
            'event_type': ['count'],
            'is_search': ['sum'],
            'server_timestamp_epoch_ms': [np.ptp],
            'label': ['max']
        }).reset_index(drop=True)
        X_train.columns = ["_".join(x) for x in X_train.columns.ravel()]
        Data.dump(X_train.drop('label_max', axis=1), f'../session_rec_sigir_data/pickle/X_train_nb{nb}.pkl')
        Data.dump(X_train['label_max'], f'../session_rec_sigir_data/pickle/y_train_nb{nb}.pkl')

    df_test = pickle_load('../session_rec_sigir_data/prepared/test.pkl')
    X_test = df_test.groupby('session_id_hash').agg({
        'event_type': ['count'],
        'is_search': ['sum'],
        'server_timestamp_epoch_ms': [np.ptp],
        'nb_after_add': ['max']
    }).reset_index()
    X_test.columns = ["_".join(x) for x in X_test.columns.ravel()]

    for nb in [0, 2, 4, 6, 8, 10]:
        print('****** Starting nb==', nb)
        Data.dump(
            X_test.query(f'nb_after_add_max=={nb}').reset_index(drop=True).drop(['session_id_hash_', 'nb_after_add_max'], axis=1),
            f'../session_rec_sigir_data/pickle/X_test_nb{nb}.pkl'
        )
        sub = X_test.query(f'nb_after_add_max=={nb}').reset_index(drop=True)[['session_id_hash_']]
        sub['label'] = np.nan
        sub.to_csv(f'../session_rec_sigir_data/prepared/sample_submission_nb{nb}.csv', index=False)
