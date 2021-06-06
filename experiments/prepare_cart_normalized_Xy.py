import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing

sys.path.append('../')
from ayniy.utils import Data
from datasets import pickle_load


if __name__ == '__main__':
    for nb in [0, 2, 4, 6, 8, 10]:
        print('****** Starting nb==', nb)
        X_train = Data.load(f'../input/pickle/X_train_nb{nb}.pkl')
        X_test = Data.load(f'../input/pickle/X_test_nb{nb}.pkl')
        y_train = Data.load(f'../input/pickle/y_train_nb{nb}.pkl')

        for c in X_train.columns:
            prep = preprocessing.QuantileTransformer()
            X_train[c] = prep.fit_transform(X_train[[c]])
            X_test[c] = prep.transform(X_test[[c]])

        Data.dump(X_train, f'../input/pickle/X_train_norm_nb{nb}.pkl')
        Data.dump(X_test, f'../input/pickle/X_test_norm_nb{nb}.pkl')
        Data.dump(y_train, f'../input/pickle/y_train_norm_nb{nb}.pkl')
