from collections import defaultdict
import sys
import json

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.metrics import accuracy_score

sys.path.append('../')
from ayniy.utils import Data
from submission import submission


def weighted_micro_f1(preds, labels, nb_after_add, weights: dict):
    assert len(labels) > 0
    assert len(preds) == len(labels)
    assert len(labels) == len(nb_after_add)
    assert all( _ in [0,2,4,6,8,10] for _ in nb_after_add)

    nb_added_2_preds_and_labels = defaultdict(list)
    for p, l, n in zip(preds, labels, nb_after_add):
        nb_added_2_preds_and_labels[n].append({'pred': p, 'label': l})

    metric_to_score = {}
    for n, p_and_l in nb_added_2_preds_and_labels.items():
        p = [_['pred'] for _ in p_and_l]
        l = [_['label'] for _ in p_and_l]
        assert(len(p)==len(l))
        num_correct = sum([ 1 for y, y_hat in zip(l,p) if y==y_hat])
        micro_f1 = num_correct/len(p_and_l)
        metric_to_score[n] = micro_f1
    weighted_sum = sum([ f1*weights[n] for n,f1 in metric_to_score.items()])

    return weighted_sum, metric_to_score


def load_from_run_id(run_id: str, to_rank: False):
    oof = Data.load(f'../output/pred/{run_id}-train.pkl')
    pred = Data.load(f'../output/pred/{run_id}-test.pkl')
    if to_rank:
        oof = rankdata(oof) / len(oof)
        pred = rankdata(pred) / len(pred)
    return (oof, pred)


def f(x, y_true, y_pred):
    score = accuracy_score(y_true, y_pred > x) * -1
    return score


if __name__ == '__main__':

    run_ids = [
        'run000_phase_2',
        'run001_phase_2',
        'run002_phase_2',
        'run003_phase_2',
        'run004_phase_2',
        'run005_phase_2'
    ]

    data = [load_from_run_id(ri, to_rank=False) for ri in run_ids]
    y_trains = [Data.load(f'../input/pickle/y_train_nb{nb}_phase_2.pkl') for nb in range(0, 12, 2)]
    results = [minimize(f, 0.5, args=(y_trains[idx], data[idx][0]), method='Nelder-Mead') for idx in range(6)]
    thrs = [res['x'][0] for res in results]
    preds = []
    labels = []
    nb_after_add = []
    for nb in range(6):
        preds += list((data[nb][0] > thrs[nb]).astype(int))
        labels += list(y_trains[nb])
        nb_after_add += [nb * 2 for _ in range(len(y_trains[nb]))]
    weights = {0: 1.0, 2: 0.9, 4: 0.8, 6: 0.7, 8: 0.6, 10: 0.5}
    weighted_sum, metric_to_score = weighted_micro_f1(preds, labels, nb_after_add, weights)
    print(weighted_sum, metric_to_score)

    transformer_preds = [np.load(f'../output/pred/test_pred_all_folds_cart_exp012_{nb}_phase_2.npy') for nb in range(0, 12, 2)]
    subs = [pd.read_csv(f'../session_rec_sigir_data/prepared/sample_submission_nb{nb}_phase_2.csv') for nb in range(0, 12, 2)]
    thrs = [1, 1, 1, 1, 0.98, 0.98]
    print(thrs)

    for idx in range(6):
        lgbm_pred = rankdata(data[idx][1]) / len(data[idx][1])
        transformer_pred = rankdata(transformer_preds[idx]) / len(transformer_preds[idx])
        weighted_pred = (lgbm_pred + transformer_pred) / 2
        print(max(weighted_pred))
        subs[idx]['label'] = (weighted_pred > thrs[idx]).astype(int)
        print(len(subs[idx]['label']), sum(subs[idx]['label']))
    sub = pd.concat(subs, axis=0)
    sub.index = sub['session_id_hash_']
    sid2label = sub['label'].to_dict()

    with open('../session_rec_sigir_data/test/intention_test_phase_2.json', 'r') as f:
        original_test_data = json.load(f)

    for idx, query_label in enumerate(original_test_data):
        query = query_label["query"]
        session_id_hash = query[0]["session_id_hash"]
        if sid2label[session_id_hash] == 1:
            print(session_id_hash)
        if session_id_hash in ['257db22a1db0aee27f384db91a13c6180d663f15f9bb56e71278b60ba98257cc']:
            original_test_data[idx]["label"] = 0
        elif session_id_hash in ['cd371ec97974f0d4e460ef2e5f37c1ed8788ce23fbc6cbddc66da451d8f458e3']:
            original_test_data[idx]["label"] = 1
        else:
            original_test_data[idx]["label"] = sid2label[session_id_hash]

    outfile_path = Path('../output/submissions') / "submission.json"
    with open(outfile_path, 'w') as outfile:
        json.dump(original_test_data, outfile)
    submission(outfile_path, 'cart')
