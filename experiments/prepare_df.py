import pandas as pd

from datasets import load_datasets, pickle_dump


if __name__ == '__main__':
    train, test, sku_to_content = load_datasets()

    pickle_dump(train, '../session_rec_sigir_data/prepared/train.pkl')
    pickle_dump(test, '../session_rec_sigir_data/prepared/test.pkl')
