import pytorch_lightning as pl
from pathlib import Path
from pytorch_lightning.loggers import WandbLogger


def get_trainer(
    config: dict,
    wandb_logger: WandbLogger = None,
    debug: bool = False
) -> pl.Trainer:
    ckpt_callback = pl.callbacks.ModelCheckpoint(
        dirpath=Path(config["file_path"]["output_dir"]) / config["exp_name"],
        filename='{epoch:03d}-{val_loss:.3f}-{val_mrr:.3f}-{val_f1_score:.3f}',
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode="min",
    )
    return pl.Trainer(
        max_epochs=config["n_epochs"] if not debug else 2,
        callbacks=[ckpt_callback, early_stop_callback],
        logger=wandb_logger,
        gpus=1,
    )
