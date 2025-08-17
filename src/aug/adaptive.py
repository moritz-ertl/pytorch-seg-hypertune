# -*- coding: utf-8 -*-

from __future__ import annotations
import numpy as np
import albumentations as A


def get_adaptive_augmentations(epoch: int, tau: float = 5.0) -> A.Compose:
    """Adaptive augmentation with epoch-dependent probability p(t).
    Uses tanh(epoch/tau) with a min floor.
    """
    p = float(max(0.05, np.tanh(epoch / max(1e-6, tau))))
    train_transform = [
        A.HorizontalFlip(p=p),
        A.VerticalFlip(p=p),
        A.RandomRotate90(p=p),
        A.Affine(scale=(0.5, 1.5), translate_percent=(0.05, 0.05), rotate=(-15, 15), p=p),
        A.PadIfNeeded(min_height=224, min_width=224),
        A.RandomCrop(height=224, width=224),
        A.OneOf([
            A.RandomBrightnessContrast(p=p),
            A.RandomGamma(p=p),
        ], p=p),
        A.OneOf([
            A.Sharpen(p=p),
            A.Blur(blur_limit=3, p=p),
        ], p=p),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation() -> A.Compose:
    return A.Compose([A.PadIfNeeded(224, 224)])