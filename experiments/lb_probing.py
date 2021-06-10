import sys
import json

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('../')
from ayniy.utils import Data
from submission import submission
from datasets import pickle_load
from prepare_cart_Xy import extract_product_action_count


if __name__ == '__main__':

    df_test = pickle_load('../session_rec_sigir_data/prepared/test.pkl')
    X_test = df_test.groupby('session_id_hash').agg({
        'is_search': ['sum'],
        'server_timestamp_epoch_ms': ['count', np.ptp],
        'nb_after_add': ['max'],
        'product_sku_hash': list,
        'product_action': list
    }).reset_index()
    X_test.columns = ["_".join(x) for x in X_test.columns.ravel()]

    for nb in [6]:
        print('****** Starting nb==', nb)
        X_test_nb = X_test.query(f'nb_after_add_max=={nb}').reset_index(drop=True)
        pos_session_ids = X_test_nb['session_id_hash_'].unique()
        assert len(pos_session_ids) > 0

    with open('../session_rec_sigir_data/test/intention_test_phase_1.json', 'r') as f:
        original_test_data = json.load(f)

    for idx, query_label in enumerate(original_test_data):
        query = query_label["query"]
        session_id_hash = query[0]["session_id_hash"]
        if session_id_hash in set(pos_session_ids):
            original_test_data[idx]["label"] = 1
        else:
            original_test_data[idx]["label"] = 0

    outfile_path = Path('../output/submissions') / "submission.json"
    with open(outfile_path, 'w') as outfile:
        json.dump(original_test_data, outfile)
    submission(outfile_path, 'cart')
