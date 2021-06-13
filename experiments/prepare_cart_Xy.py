import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing

sys.path.append('../')
from ayniy.utils import Data
from datasets import pickle_load


def extract_product_action_count(df, nb):
    _df = df.copy()
    _df['target_product_sku_hash'] = _df['product_sku_hash_list'].map(lambda x: x[len(x) - nb - 1])
    _df['product_action_list_nb'] = _df['product_action_list'].map(lambda x: x[-nb:])
    _df['product_sku_hash_list_nb'] = _df['product_sku_hash_list'].map(lambda x: x[-nb:])
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
        'num_add_not_same_product_nb',
        'num_detail_not_same_product_nb',
        'num_remove_not_same_product_nb',
    ]]


def extract_timestamp(df):
    _df = df.copy()
    _df = _df.groupby('session_id_hash')['server_timestamp'].max().reset_index()
    _df['server_timestamp_day'] = _df['server_timestamp'].dt.day
    _df['server_timestamp_dow'] = _df['server_timestamp'].dt.weekday
    _df['server_timestamp_hour'] = _df['server_timestamp'].dt.hour
    return _df[[
        'server_timestamp_day',
        'server_timestamp_dow',
        'server_timestamp_hour'
    ]]


def extract_product(X_train, X_test_nb, nb):
    sku_to_content = pd.read_pickle('../session_rec_sigir_data/prepared/sku_to_content.pkl')
    _X_train = X_train.copy()
    _X_test_nb = X_test_nb.copy()

    _df = pd.concat([_X_train, X_test_nb], axis=0)
    _df['product_sku_hash'] = _df['product_sku_hash_list'].map(lambda x: x[len(x) - nb - 1])
    _df = pd.merge(_df['product_sku_hash'], sku_to_content, on='product_sku_hash', how='left')
    for c in ['category_hash',
              'category_hash_first_level',
              'category_hash_second_level',
              'category_hash_third_level']:
        le = preprocessing.LabelEncoder()
        _df[c] = le.fit_transform(_df[c].fillna('unkown'))
    _df = pd.concat([
        _df[['category_hash',
             'category_hash_first_level',
             'category_hash_second_level',
             'category_hash_third_level',
             'price_bucket']],
        pd.DataFrame(
            _df['description_vector'].map(lambda x: x if isinstance(x, list) else [0 for _ in range(50)]).to_list(),
            columns=[f'desc_{i}' for i in range(50)]
        ),
        pd.DataFrame(
            _df['image_vector'].map(lambda x: x if isinstance(x, list) else [0 for _ in range(50)]).to_list(),
            columns=[f'img_{i}' for i in range(50)]
        )
    ], axis=1)
    return _df


if __name__ == '__main__':

    df_test = pickle_load('../session_rec_sigir_data/prepared/test_phase_2.pkl')
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
        X_train = pd.concat([X_train, extract_timestamp(df_train)], axis=1)

        X_test_nb = X_test.query(f'nb_after_add_max=={nb}').reset_index(drop=True)
        X_test_nb = pd.concat([X_test_nb, extract_product_action_count(X_test_nb, nb)], axis=1)
        X_test_nb = pd.concat([X_test_nb, extract_timestamp(df_test.query(f'nb_after_add=={nb}'))], axis=1)

        _df = extract_product(X_train, X_test_nb, nb)
        X_train = pd.concat([X_train, _df[:len(X_train)].reset_index(drop=True)], axis=1)
        X_test_nb = pd.concat([X_test_nb, _df[len(X_train):].reset_index(drop=True)], axis=1)

        print(X_train.drop(['product_sku_hash_list', 'product_action_list', 'label_max'], axis=1).shape)
        Data.dump(X_train.drop(['product_sku_hash_list', 'product_action_list', 'label_max'], axis=1),
                  f'../input/pickle/X_train_nb{nb}_phase_2.pkl')
        Data.dump(X_train['label_max'], f'../input/pickle/y_train_nb{nb}_phase_2.pkl')
        Data.dump(
            X_test_nb.drop([
                'session_id_hash_',
                'nb_after_add_max',
                'product_sku_hash_list',
                'product_action_list'
            ], axis=1),
            f'../input/pickle/X_test_nb{nb}_phase_2.pkl'
        )
        sub = X_test_nb[['session_id_hash_']].copy()
        sub.loc[:, 'label'] = np.nan
        sub.to_csv(f'../session_rec_sigir_data/prepared/sample_submission_nb{nb}_phase_2.csv', index=False)
