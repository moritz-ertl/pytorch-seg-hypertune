# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 22:46:34 2025

@author: Moritz Ertl
"""

from __future__ import annotations
import argparse, yaml
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from src.data.datamodule import SegDataModule
from src.models.lightning_module import MulticlassSeg


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config_path: str):
    cfg = load_cfg(config_path)
    seed_everything(cfg.get("seed", 17), workers=True)

    dm = SegDataModule(cfg["data"], cfg["dataloader"], cfg.get("aug", {}))
    model = MulticlassSeg(**cfg["model"])

    callbacks = [
        ModelCheckpoint(monitor="valid_combined_metric", mode="max", save_top_k=1, dirpath="checkpoints"),
        EarlyStopping(monitor="valid_combined_metric", mode="max", patience=5),
        LearningRateMonitor(logging_interval="epoch"),
    ]
    logger = TensorBoardLogger(save_dir="tb_logs", name="run")

    trainer = Trainer(logger=logger, callbacks=callbacks, **cfg["trainer"])
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment_unet.yaml")
    args = ap.parse_args()
    main(args.config)
