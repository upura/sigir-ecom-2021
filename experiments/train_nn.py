import json
import random
import sys
import yaml

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

sys.path.append('../')
from ayniy.utils import Data
from datasets import pickle_load
from src.utils import seed_everything
from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.dataset import RecTaskDataModule
from src.plmodel import RecTaskPLModel
from src.trainer import get_trainer
# from src.submission import submission


rand = random.randint(0, 100000)

def run(config: dict, debug: bool, holdout: bool) -> None:

    pl.seed_everything(777)

    with open(config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
        config["exp_name"] = Path(f.name).stem

    for nb in [0, 2, 4, 6, 8, 10]:
        print('****** Starting nb==', nb)

        test = pickle_load('../session_rec_sigir_data/prepared/test.pkl')
        test = test.query(f'nb_after_add=={nb}').reset_index(drop=True)
        test_session_ids = set(test["session_id_hash"].unique())
        sku_to_content = pd.read_pickle('../session_rec_sigir_data/prepared/sku_to_content.pkl')

        df_pos = pd.read_pickle(f'../session_rec_sigir_data/prepared/train_pos_nb{nb}.pkl')
        df_neg = pd.read_pickle(f'../session_rec_sigir_data/prepared/train_neg_nb{nb}.pkl')

        df_pos['label'] = 1
        df_neg['label'] = 0
        train = pd.concat([df_pos, df_neg], axis=0)

        print(f"train: {train.shape}")
        print(f"test: {test.shape}")
        print(f"sku_to_content: {sku_to_content.shape}")
        print(f"number of test sessions: {len(test_session_ids)}")

        pr = Preprocessor(config)
        train_preprocessed, test_preprocessed= pr.run(train, test, sku_to_content)
        del train, test, sku_to_content
        print(f"train_preprocessed: {train_preprocessed.shape}")
        print(f"test_preprocessed: {test_preprocessed.shape}")

        train_session_info = train_preprocessed.groupby("session_id_hash").agg({"product_sku_hash": ["nunique"], "label": ["max"]}).reset_index()
        train_session_info.columns = ["session_id_hash", "n_items", "label"]
        cv = StratifiedKFold(**config["fold_params"])
        folds = cv.split(
            train_session_info,
            pd.cut(
                train_session_info["n_items"],
                config["fold_params"]["n_splits"],
                labels=False,
            ),
        )

        num_labels = len(pr.index_to_label_dict["product_sku_hash"]) + 1   # plus padding id
        test_preprocessed["label"] = -1
        test_session_seqs = pr.get_session_sequences(test_preprocessed)

        test_pred_all_folds = np.zeros((len(test_session_seqs), 1), dtype=np.float32)
        print(f"number of preprocessed test sessions: {len(test_session_seqs)}")
        print(f"ratio of preprocessed test sessions: {len(test_session_seqs) / len(test_session_ids)}")

        for i_fold, (trn_idx, val_idx) in enumerate(folds):
            if holdout and i_fold > 0:
                break

            if config["wandb"]["use"] and not debug:
                wandb.init(
                    name=f"{config['exp_name']}-fold-{i_fold}-{rand}",
                    project=config["wandb"]["project"],
                    entity=config["wandb"]["entity"],
                    tags=config["wandb"]["tags"] + [config["exp_name"]] + ["debug" if debug else "prod"],
                    reinit=True,
                )
                wandb_logger = WandbLogger(
                    name=f"{config['exp_name']}-fold-{i_fold}-{rand}",
                    project=config["wandb"]["project"],
                    tags=config["wandb"]["tags"] + [config["exp_name"]] + ["debug" if debug else "prod"],
                )
                wandb_logger.log_hyperparams(dict(config))
                wandb_logger.log_hyperparams({
                    "fold": i_fold,
                })
            else:
                wandb_logger = None

            train_session_ids = train_session_info.iloc[trn_idx]["session_id_hash"].tolist()
            val_session_ids = train_session_info.iloc[val_idx]["session_id_hash"].tolist()
            train_session_seqs = pr.get_session_sequences(
                train_preprocessed[train_preprocessed["session_id_hash"].isin(train_session_ids)]
            )
            val_session_seqs = pr.get_session_sequences(
                train_preprocessed[train_preprocessed["session_id_hash"].isin(val_session_ids)]
            )
            print(f"number of train sessions: {len(train_session_seqs)}")
            print(f"number of valid sessions: {len(val_session_seqs)}")

            dataset = RecTaskDataModule(
                config,
                train_session_seqs,
                val_session_seqs,
                test_session_seqs,
                num_labels,
            )
            model = RecTaskPLModel(config, num_labels=num_labels)
            trainer = get_trainer(config, wandb_logger=wandb_logger, debug=debug)
            trainer.fit(model, dataset)
            best_ckpt = (
                Path(config["file_path"]["output_dir"])
                / config["exp_name"]
                / f"best_model_fold{i_fold}.ckpt"
            )
            trainer.save_checkpoint(best_ckpt)

            model.to(torch.device("cpu"))
            test_dataloader = dataset.test_dataloader()
            y_pred_list = []
            for x_batch in test_dataloader:
                y_pred = model.forward(x_batch, torch.device("cpu"))
                y_pred = y_pred.detach().numpy()
                y_pred = y_pred.reshape(-1)
                y_pred_list.append(y_pred)
            test_pred = np.array(y_pred_list)

            if holdout:
                test_pred_all_folds += test_pred.reshape(-1, 1)
            else:
                test_pred_all_folds += test_pred.reshape(-1, 1) / config["fold_params"]["n_splits"]
            np.save(f'../output/pred/test_pred_all_folds_{config["exp_name"]}_{nb}', test_pred_all_folds)


if __name__ == '__main__':
    run(config='configs/cart_exp012.yml',
        debug=False,
        holdout=True)
