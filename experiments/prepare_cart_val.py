import sys

import pandas as pd

sys.path.append('../')
from ayniy.utils import Data
from datasets import pickle_load
from weighted_averaging import weighted_micro_f1


MODE = 'adversarial'

if __name__ == '__main__':
    nb = 0
    preds = []
    labels = []
    nb_after_add = []
    prop = [8, 8, 8, 8, 8, 8]

    for run_id in ['run008', 'run009', 'run010', 'run011', 'run012', 'run013']:
        print('****** Starting run_id==', run_id)
        train_adv = Data.load(f'../output/pred/{run_id}-train.pkl')
        X_train = Data.load(f'../input/pickle/X_train_nb{nb * 2}.pkl')
        X_test = Data.load(f'../input/pickle/X_test_nb{nb * 2}.pkl')

        X_train['label'] = Data.load(f'../input/pickle/y_train_nb{nb * 2}.pkl')
        X_train['adv'] = train_adv[:len(X_train)]
        X_train['pred'] = Data.load(f'../output/pred/run00{nb}-train.pkl')
        X_train['pred'] = (X_train['pred'] > 0.5).astype(int)
        X_train = X_train.sort_values('adv', ascending=False)
        X_train_neg = X_train.query('label==0')
        X_train_pos = X_train.query('label==1')

        if MODE == 'all':
            preds += X_train_neg['pred'].to_list()
            labels += X_train_neg['label'].to_list()
            preds += X_train_pos['pred'].to_list()
            labels += X_train_pos['label'].to_list()
            nb_after_add += [nb * 2 for _ in range(len(X_train_neg) + len(X_train_pos))]
        elif MODE == 'adversarial':
            preds += X_train_neg['pred'][:int(len(X_test) * prop[nb] // 10)].to_list()
            labels += X_train_neg['label'][:int(len(X_test) * prop[nb] // 10)].to_list()
            preds += X_train_pos['pred'][:int(len(X_test) * (10 - prop[nb]) // 10)].to_list()
            labels += X_train_pos['label'][:int(len(X_test) * (10 - prop[nb]) // 10)].to_list()
            nb_after_add += [nb * 2 for _ in range(
                int(len(X_test) * prop[nb] // 10) + int(len(X_test) * (10 - prop[nb]) // 10)
            )]
        elif MODE == 'random':
            X_train_neg = X_train_neg.sample(n=int(len(X_test) * prop[nb] // 10), random_state=123).reset_index(drop=True)
            X_train_pos = X_train_pos.sample(n=int(len(X_test) * (10 - prop[nb]) // 10), random_state=123).reset_index(drop=True)
            preds += X_train_neg['pred'].to_list()
            labels += X_train_neg['label'].to_list()
            preds += X_train_pos['pred'].to_list()
            labels += X_train_pos['label'].to_list()
            nb_after_add += [nb * 2 for _ in range(len(X_train_neg) + len(X_train_pos))]

        nb += 1

    weights = {0: 1.0, 2: 0.9, 4: 0.8, 6: 0.7, 8: 0.6, 10: 0.5}
    weighted_sum, metric_to_score = weighted_micro_f1(preds, labels, nb_after_add, weights)
    print(weighted_sum, metric_to_score)
