import sys

import numpy as np
import pandas as pd

sys.path.append('../')
from ayniy.utils import Data
from datasets import pickle_load


def extract_product_action_count(df, nb):
    _df = df.copy()
    _df['target_product_sku_hash'] = _df['product_sku_hash_list'].map(lambda x: x[len(x) - nb - 1])
    _df['product_action_list_nb'] = _df['product_action_list'].map(lambda x: x[len(x) - nb:])
    _df['product_sku_hash_list_nb'] = _df['product_sku_hash_list'].map(lambda x: x[len(x) - nb:])
    _df['same_product_sku_hash'] = _df.apply(
        lambda x: [product is x['target_product_sku_hash'] for product in x['product_sku_hash_list'] if isinstance(product, str)],
        axis=1
    )
    _df['product_action_list'] = _df['product_action_list'].map(lambda x: [action for action in x if isinstance(action, str)])

    for pc in ['add', 'detail', 'remove']:
        _df[f'num_{pc}_same_product'] = _df.apply(
            lambda x: len([(action, product) for action, product in zip(x['product_action_list'], x['same_product_sku_hash']) if (action==pc) and product]),
            axis=1
        )
        _df[f'num_{pc}_not_same_product'] = _df.apply(
            lambda x: len([(action, product) for action, product in zip(x['product_action_list'], x['same_product_sku_hash']) if (action==pc) and not(product)]),
            axis=1
        )

    if nb == 0:
        return _df[[
            'num_add_same_product',
            'num_detail_same_product',
            'num_remove_same_product',
            'num_add_not_same_product',
            'num_detail_not_same_product',
            'num_remove_not_same_product',
        ]]

    _df['same_product_sku_hash_nb'] = _df.apply(
        lambda x: [product is x['target_product_sku_hash'] for product in x['product_sku_hash_list_nb'] if isinstance(product, str)],
        axis=1
    )
    _df['product_action_list_nb'] = _df['product_action_list_nb'].map(lambda x: [action for action in x if isinstance(action, str)])

    for pc in ['add', 'detail', 'remove']:
        _df[f'num_{pc}_same_product_nb'] = _df.apply(
            lambda x: len([(action, product) for action, product in zip(x['product_action_list_nb'], x['same_product_sku_hash_nb']) if (action==pc) and product]),
            axis=1
        )
        _df[f'num_{pc}_not_same_product_nb'] = _df.apply(
            lambda x: len([(action, product) for action, product in zip(x['product_action_list_nb'], x['same_product_sku_hash_nb']) if (action==pc) and not(product)]),
            axis=1
        )

    return _df[[
        'num_add_same_product',
        'num_detail_same_product',
        'num_remove_same_product',
        'num_add_not_same_product',
        'num_detail_not_same_product',
        'num_remove_not_same_product',
        'num_add_same_product_nb',
        'num_detail_same_product_nb',
        'num_remove_same_product_nb',
        'num_add_not_same_product_nb',
        'num_detail_not_same_product_nb',
        'num_remove_not_same_product_nb',
    ]]


if __name__ == '__main__':
    for nb in [0, 2, 4, 6, 8, 10]:
        print('****** Starting nb==', nb)
        df_pos = pd.read_pickle(f'../session_rec_sigir_data/prepared/train_pos_nb{nb}.pkl')
        df_neg = pd.read_pickle(f'../session_rec_sigir_data/prepared/train_neg_nb{nb}.pkl')

        df_pos['label'] = 1
        df_neg['label'] = 0
        df_train = pd.concat([df_pos, df_neg], axis=0)
        X_train = df_train.groupby('session_id_hash').agg({
            'is_search': ['sum'],
            'server_timestamp_epoch_ms': ['count', np.ptp],
            'label': ['max'],
            'product_sku_hash': list,
            'product_action': list
        }).reset_index(drop=True)
        X_train.columns = ["_".join(x) for x in X_train.columns.ravel()]
        X_train = pd.concat([X_train, extract_product_action_count(X_train, nb)], axis=1)
        X_train = X_train.drop(['product_sku_hash_list', 'product_action_list'], axis=1)
        Data.dump(X_train.drop('label_max', axis=1), f'../input/pickle/X_train_nb{nb}.pkl')
        Data.dump(X_train['label_max'], f'../input/pickle/y_train_nb{nb}.pkl')

    df_test = pickle_load('../session_rec_sigir_data/prepared/test.pkl')
    X_test = df_test.groupby('session_id_hash').agg({
        'is_search': ['sum'],
        'server_timestamp_epoch_ms': ['count', np.ptp],
        'nb_after_add': ['max'],
        'product_sku_hash': list,
        'product_action': list
    }).reset_index()
    X_test.columns = ["_".join(x) for x in X_test.columns.ravel()]

    for nb in [0, 2, 4, 6, 8, 10]:
        print('****** Starting nb==', nb)
        X_test_nb = X_test.query(f'nb_after_add_max=={nb}').reset_index(drop=True)
        X_test_nb = pd.concat([X_test_nb, extract_product_action_count(X_test_nb, nb)], axis=1)

        Data.dump(
            X_test_nb.drop([
                'session_id_hash_',
                'nb_after_add_max',
                'product_sku_hash_list',
                'product_action_list'
            ], axis=1),
            f'../input/pickle/X_test_nb{nb}.pkl'
        )
        sub = X_test_nb[['session_id_hash_']]
        sub['label'] = np.nan
        sub.to_csv(f'../session_rec_sigir_data/prepared/sample_submission_nb{nb}.csv', index=False)
