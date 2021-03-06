import pandas as pd
from tqdm import tqdm

from datasets import pickle_load


if __name__ == '__main__':
    train = pickle_load('../session_rec_sigir_data/prepared/train.pkl')

    session_id_hash_contain_purchase = train.query('product_action=="purchase"')['session_id_hash'].unique()
    session_id_hash_contain_add = train.query('product_action=="add"')['session_id_hash'].unique()
    session_id_hash_pos= set(session_id_hash_contain_purchase) & set(session_id_hash_contain_add)
    session_id_hash_neg = set(session_id_hash_contain_add) - set(session_id_hash_contain_purchase)

    train_pos = train[train['session_id_hash'].isin(session_id_hash_pos)]
    train_pos_gpb = train_pos.groupby('session_id_hash')
    for nb in [0, 2, 4, 6, 8, 10]:
        print('****** Starting nb==', nb)
        dfs = []
        for tpg in tqdm(train_pos_gpb):
            sid, tmp_train = tpg
            tmp_train = tmp_train.reset_index(drop=True)
            try:
                purchased_product_sku_hash = tmp_train.query('product_action=="purchase"')['product_sku_hash'].values[0]
                purchase_idx_purchased_product_sku_hash = tmp_train.query('product_action=="purchase"').index[0]
                ac_idx_purchased_product_sku_hash = tmp_train.query(f'product_action=="add" and product_sku_hash=="{purchased_product_sku_hash }"').index[0]
                if ((purchase_idx_purchased_product_sku_hash > ac_idx_purchased_product_sku_hash + nb)
                        and (ac_idx_purchased_product_sku_hash + nb < len(tmp_train))
                        and (purchased_product_sku_hash not in tmp_train['product_sku_hash'][ac_idx_purchased_product_sku_hash + 1: ac_idx_purchased_product_sku_hash  + nb + 1].values)):
                    dfs.append(tmp_train.loc[:ac_idx_purchased_product_sku_hash + nb])
            except IndexError:
                pass
        pd.concat(dfs, axis=0).reset_index(drop=True).to_pickle(f'../session_rec_sigir_data/prepared/train_pos_nb{nb}.pkl')

    train_neg = train[train['session_id_hash'].isin(session_id_hash_neg)]
    train_neg_gpb = train_neg.groupby('session_id_hash')
    for nb in [0, 2, 4, 6, 8, 10]:
        print('****** Starting nb==', nb)
        dfs = []
        for tng in tqdm(train_neg_gpb):
            sid, tmp_train = tng
            tmp_train = tmp_train.reset_index(drop=True)
            try:
                ac_idx = tmp_train.query('product_action=="add"').index[-1]
                added_product_sku_hash = tmp_train.query('product_action=="add"')['product_sku_hash'].values[-1]
                if ((ac_idx + nb < len(tmp_train))
                        and (added_product_sku_hash not in tmp_train['product_sku_hash'][ac_idx + 1: ac_idx + nb + 1].values)):
                    dfs.append(tmp_train.loc[:ac_idx + nb])
            except IndexError:
                pass
        pd.concat(dfs, axis=0).reset_index(drop=True).to_pickle(f'../session_rec_sigir_data/prepared/train_neg_nb{nb}.pkl')
