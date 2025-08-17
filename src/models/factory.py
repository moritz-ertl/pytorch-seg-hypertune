# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 22:32:24 2025

@author: Moritz Ertl
"""

from __future__ import annotations
import segmentation_models_pytorch as smp

def create_smp_model(arch: str, encoder_name: str, classes: int, **kwargs):
    return smp.create_model(
        arch,
        encoder_name=encoder_name,
        encoder_weights=kwargs.pop("encoder_weights", "imagenet"),
        in_channels=kwargs.pop("in_channels", 3),
        classes=classes,
        **kwargs,
    )
