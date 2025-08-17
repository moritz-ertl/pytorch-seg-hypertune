# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 22:33:27 2025

@author: Moritz Ertl
"""

from __future__ import annotations
from typing import Dict, Any
import optuna

def _suggest(trial: optuna.trial.Trial, spec: Dict[str, Any]):
    t = spec.get("type")
    if t == "categorical":
        return trial.suggest_categorical(spec.get("name", "_"), spec["choices"])  # name will be overwritten
    elif t == "float":
        low, high = float(spec["low"]), float(spec["high"])
        step = spec.get("step")
        log = bool(spec.get("log", False))
        if step is not None:
            return trial.suggest_float(spec.get("name", "_"), low, high, step=float(step), log=log)
        return trial.suggest_float(spec.get("name", "_"), low, high, log=log)
    elif t == "int":
        low, high = int(spec["low"]), int(spec["high"])
        step = int(spec.get("step", 1))
        return trial.suggest_int(spec.get("name", "_"), low, high, step=step)
    else:
        raise ValueError(f"Unsupported param type: {t}")

def _set_nested(cfg: Dict[str, Any], dotted_key: str, value: Any):
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value

def apply_sampled_params(trial: optuna.trial.Trial, cfg: Dict[str, Any], space: Dict[str, Any]) -> None:
    for dotted_key, spec in space.items():
        spec = dict(spec)
        spec["name"] = dotted_key
        val = _suggest(trial, spec)
        _set_nested(cfg, dotted_key, val)