# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 22:47:07 2025

@author: Moritz Ertl
"""

from __future__ import annotations
import argparse, yaml, optuna
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from optuna.integration import PyTorchLightningPruningCallback
from src.data.datamodule import SegDataModule
from src.models.lightning_module import MulticlassSeg
from src.utils.search_space import apply_sampled_params


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def objective(trial: optuna.trial.Trial, base_cfg: dict) -> float:
    cfg = yaml.safe_load(yaml.dump(base_cfg))
    space = cfg.get("optuna", {}).get("search_space", {})
    apply_sampled_params(trial, cfg, space)

    seed_everything(cfg.get("seed", 17), workers=True)

    dm = SegDataModule(cfg["data"], cfg["dataloader"], cfg.get("aug", {}))
    model = MulticlassSeg(**cfg["model"])

    logger = TensorBoardLogger(save_dir="tb_logs", name=f"trial_{trial.number}")
    callbacks = [
        PyTorchLightningPruningCallback(trial, monitor="valid_combined_metric"),
        EarlyStopping(monitor="valid_combined_metric", mode="max", patience=5),
        ModelCheckpoint(monitor="valid_combined_metric", mode="max", save_top_k=1, dirpath="checkpoints"),
    ]

    trainer = Trainer(logger=logger, callbacks=callbacks, **cfg["trainer"])
    trainer.fit(model, datamodule=dm)

    metric = trainer.callback_metrics.get("valid_combined_metric", None)
    return float(metric.item()) if metric is not None else 0.0


def main(config_path: str, trials: int | None):
    cfg = load_cfg(config_path)
    study = optuna.create_study(direction=cfg["optuna"]["direction"])
    study.optimize(lambda tr: objective(tr, cfg), n_trials=trials or cfg["optuna"]["n_trials"])
    print("Best value:", study.best_value)
    print("Best params:", study.best_params)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--trials", type=int, default=None)
    args = ap.parse_args()
    main(args.config, args.trials)