import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('../')
from ayniy.utils import Data
from datasets import pickle_load


def extract_features(sample_df, nb):
    target_product_sku_hash = sample_df.loc[len(sample_df) - nb - 1]['product_sku_hash']
    sample_df_event_product = sample_df.query('event_type=="event_product"').reset_index(drop=True)
    sample_df_event_product.loc[:, 'product_sku_hash'] = (sample_df_event_product['product_sku_hash'] == target_product_sku_hash).astype(int).values
    num_add_same_product = len(sample_df_event_product.query('product_action=="add" and product_sku_hash==1'))
    num_detail_same_product = len(sample_df_event_product.query('product_action=="detail" and product_sku_hash==1'))
    num_remove_same_product = len(sample_df_event_product.query('product_action=="remove" and product_sku_hash==1'))
    num_add_not_same_product = len(sample_df_event_product.query('product_action=="add" and product_sku_hash==0'))
    num_detail_not_same_product = len(sample_df_event_product.query('product_action=="detail" and product_sku_hash==0'))
    num_remove_not_same_product = len(sample_df_event_product.query('product_action=="remove" and product_sku_hash==0'))
    if nb == 0:
        return (
            num_add_same_product,
            num_detail_same_product,
            num_remove_same_product,
            num_add_not_same_product,
            num_detail_not_same_product,
            num_remove_not_same_product,
        )
    num_add_same_product_nb = len(sample_df_event_product.query(f'product_action=="add" and product_sku_hash==1 and index>={len(sample_df) - nb}'))
    num_detail_same_product_nb = len(sample_df_event_product.query(f'product_action=="detail" and product_sku_hash==1 and index>={len(sample_df) - nb}'))
    num_remove_same_product_nb = len(sample_df_event_product.query(f'product_action=="remove" and product_sku_hash==1 and index>={len(sample_df) - nb}'))
    num_add_not_same_product_nb = len(sample_df_event_product.query(f'product_action=="add" and product_sku_hash==0 and index>={len(sample_df) - nb}'))
    num_detail_not_same_product_nb = len(sample_df_event_product.query(f'product_action=="detail" and product_sku_hash==0 and index>={len(sample_df) - nb}'))
    num_remove_not_same_product_nb = len(sample_df_event_product.query(f'product_action=="remove" and product_sku_hash==0 and index>={len(sample_df) - nb}'))
    return (
        num_add_same_product,
        num_detail_same_product,
        num_remove_same_product,
        num_add_not_same_product,
        num_detail_not_same_product,
        num_remove_not_same_product,
        num_add_same_product_nb,
        num_detail_same_product_nb,
        num_remove_same_product_nb,
        num_add_not_same_product_nb,
        num_detail_not_same_product_nb,
        num_remove_not_same_product_nb,
    )


colnames0 = [
    'num_add_same_product',
    'num_detail_same_product',
    'num_remove_same_product',
    'num_add_not_same_product',
    'num_detail_not_same_product',
    'num_remove_not_same_product',
]
colnames = [
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
]


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
            'label': ['max']
        }).reset_index(drop=True)
        X_train.columns = ["_".join(x) for x in X_train.columns.ravel()]

        train_gpb = df_train.groupby('session_id_hash')
        features = []
        for tpg in tqdm(train_gpb):
            sid, tmp_train = tpg
            tmp_train = tmp_train.reset_index(drop=True)
            features.append(extract_features(tmp_train, nb))

        if nb == 0:
            features = pd.DataFrame(features, columns=colnames0)
        else:
            features = pd.DataFrame(features, columns=colnames)
        X_train = pd.concat([X_train, features], axis=1)
        Data.dump(X_train.drop('label_max', axis=1), f'../input/pickle/X_train_nb{nb}.pkl')
        Data.dump(X_train['label_max'], f'../input/pickle/y_train_nb{nb}.pkl')

    df_test = pickle_load('../session_rec_sigir_data/prepared/test.pkl')
    X_test = df_test.groupby('session_id_hash').agg({
        'is_search': ['sum'],
        'server_timestamp_epoch_ms': ['count', np.ptp],
        'nb_after_add': ['max']
    }).reset_index()
    X_test.columns = ["_".join(x) for x in X_test.columns.ravel()]

    for nb in [0, 2, 4, 6, 8, 10]:
        print('****** Starting nb==', nb)
        X_test_nb = X_test.query(f'nb_after_add_max=={nb}').reset_index(drop=True)
        test_gb = df_test.query(f'nb_after_add=={nb}').groupby('session_id_hash')
        features = []
        for tpg in tqdm(test_gb):
            sid, tmp_test = tpg
            tmp_test = tmp_test.reset_index(drop=True)
            features.append(extract_features(tmp_test, nb))

        if nb == 0:
            features = pd.DataFrame(features, columns=colnames0)
        else:
            features = pd.DataFrame(features, columns=colnames)
        X_test_nb = pd.concat([X_test_nb, features], axis=1)

        Data.dump(
            X_test_nb.drop(['session_id_hash_', 'nb_after_add_max'], axis=1),
            f'../input/pickle/X_test_nb{nb}.pkl'
        )
        sub = X_test_nb[['session_id_hash_']]
        sub['label'] = np.nan
        sub.to_csv(f'../session_rec_sigir_data/prepared/sample_submission_nb{nb}.csv', index=False)
