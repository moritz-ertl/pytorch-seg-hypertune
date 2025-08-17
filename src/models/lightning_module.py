# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 23:01:47 2025

@author: Moritz Ertl
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
from lightning import LightningModule
import segmentation_models_pytorch as smp
from torch.optim import Adam
from .factory import create_smp_model

class MulticlassSeg(LightningModule):
    def __init__(self, arch: str, encoder_name: str, classes: int, lr: float = 2e-4, loss: str = "DiceLoss",
                 weight_decay: float = 1e-6, **dec_kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = create_smp_model(arch, encoder_name, classes, **dec_kwargs)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.num_classes = int(classes)
        if loss == "DiceLoss":
            self.loss_fn = smp.losses.DiceLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        elif loss == "JaccardLoss":
            self.loss_fn = smp.losses.JaccardLoss(smp.losses.MULTICLASS_MODE, from_logits=True)
        elif loss == "FocalLoss":
            self.loss_fn = smp.losses.FocalLoss(smp.losses.MULTICLASS_MODE)
        else:
            raise ValueError(f"Unsupported loss function: {loss}")
        self._ep_out = {"train": [], "valid": [], "test": []}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        y = y.long()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        pred = logits.softmax(1).argmax(1)
        tp, fp, fn, tn = smp.metrics.get_stats(pred, y, mode="multiclass", num_classes=self.num_classes)
        self._ep_out[stage].append({"loss": loss, "tp": tp, "fp": fp, "fn": fn})
        self.log(f"{stage}_loss", loss, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "valid")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    @torch.no_grad()
    def _epoch_end(self, stage: str):
        outs = self._ep_out[stage]
        if not outs:
            return
        tp = torch.cat([o["tp"] for o in outs], 0).sum(0)
        fp = torch.cat([o["fp"] for o in outs], 0).sum(0)
        fn = torch.cat([o["fn"] for o in outs], 0).sum(0)
        iou_per_class = tp / (tp + fp + fn + 1e-6)
        class_pix = tp + fn
        weighted_iou = (iou_per_class * class_pix).sum() / (class_pix.sum() + 1e-6)
        dataset_iou = tp.sum() / (tp.sum() + fp.sum() + fn.sum() + 1e-6)
        alpha = 0.7
        combined = alpha * iou_per_class[2] + (1 - alpha) * weighted_iou
        for i, v in enumerate(iou_per_class):
            self.log(f"{stage}_class_{i}_iou", v, prog_bar=(i == 2))
        self.log(f"{stage}_weighted_iou", weighted_iou, prog_bar=True)
        self.log(f"{stage}_dataset_iou", dataset_iou)
        self.log(f"{stage}_combined_metric", combined, prog_bar=True)
        self._ep_out[stage].clear()

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("valid")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def configure_optimizers(self):
        opt = Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.1, patience=5, min_lr=1e-6, verbose=True)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "valid_combined_metric"}}