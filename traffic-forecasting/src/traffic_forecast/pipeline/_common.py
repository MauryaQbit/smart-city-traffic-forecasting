"""Shared helpers for the pipeline stages."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from traffic_forecast import config
from traffic_forecast.features import (
    chronological_split,
    engineer_features,
    load_raw,
)


def ensure_dirs() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_and_split():
    """Load raw, engineer features on the full frame, split chronologically,
    persist the pickled train/val frames for downstream stages."""
    ensure_dirs()
    raw = load_raw(config.RAW_PATH)
    feat = engineer_features(raw)
    train_df, val_df = chronological_split(feat, val_frac=config.VAL_FRAC)
    train_df = train_df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    val_df = val_df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    train_df.to_pickle(config.TRAIN_PKL)
    val_df.to_pickle(config.VAL_PKL)
    return train_df, val_df


def load_pickled_split():
    return pd.read_pickle(config.TRAIN_PKL), pd.read_pickle(config.VAL_PKL)


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
