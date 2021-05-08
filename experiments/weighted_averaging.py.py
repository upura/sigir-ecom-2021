import sys
import json

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import f1_score

sys.path.append('../')
from ayniy.utils import Data
from submission import submission


def load_from_run_id(run_id: str, to_rank: False):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if to_rank:
        oof = rankdata(oof) / len(oof)
        pred = rankdata(pred) / len(pred)
    return (oof, pred)


def f(x, y_true, y_pred):
    score = f1_score(y_true, y_pred > x) * -1
    return score


if __name__ == '__main__':

    run_ids = [
        'run000',
        'run001',
        'run002',
        'run003',
        'run004',
        'run005'
    ]

    data = [load_from_run_id(ri, to_rank=False) for ri in run_ids]
    y_trains = [Data.load(f'../input/pickle/y_train_nb{nb}.pkl') for nb in range(0, 12, 2)]
    results = [minimize(f, 0.5, args=(y_trains[idx], data[idx][0]), method='Nelder-Mead') for idx in range(6)]
    thrs = [res['x'][0] for res in results]
    scores = [-1 * res['fun'] for res in results]
    cv_score = scores[0] + scores[1] * 0.9 + scores[2] * 0.8 + scores[3] * 0.7 + scores[4] * 0.6 + scores[5] * 0.5
    print(cv_score, scores)

    subs = [pd.read_csv(f'../session_rec_sigir_data/prepared/sample_submission_nb{nb}.csv') for nb in range(0, 12, 2)]
    for idx in range(6):
        subs[idx]['label'] = (data[idx][1] > thrs[idx]).astype(int)
    sub = pd.concat(subs, axis=0)
    sub.index = sub['session_id_hash_']
    sid2label = sub['label'].to_dict()

    with open('../session_rec_sigir_data/test/intention_test_phase_1.json', 'r') as f:
        original_test_data = json.load(f)

    for idx, query_label in enumerate(original_test_data):
        query = query_label["query"]
        session_id_hash = query[0]["session_id_hash"]
        # 要検証
        original_test_data[idx]["label"] = int(1 - sid2label[session_id_hash])

    outfile_path = Path('../output/submissions') / "submission.json"
    with open(outfile_path, 'w') as outfile:
        json.dump(original_test_data, outfile)
    submission(outfile_path, 'cart')
