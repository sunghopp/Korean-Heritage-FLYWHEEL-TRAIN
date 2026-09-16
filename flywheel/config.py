from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    project_id: str
    bucket: str
    text_prefix: str
    audio_prefix: str
    flywheel_prefix: str
    base_model: str
    production_adapter_uri: str
    replay_manifest: str
    old_golden_manifest: str
    new_holdout_ratio: float
    replay_ratio: float
    min_samples: int
    seed: int
    old_wer_regression_max_pp: float
    new_wer_improvement_min_pp: float


def load_config(path: str) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gate = raw["promotion"]
    return Config(
        project_id=str(raw["project_id"]), bucket=raw["bucket"],
        text_prefix=raw["text_prefix"].strip("/"),
        audio_prefix=raw["audio_prefix"].strip("/"),
        flywheel_prefix=raw["flywheel_prefix"].strip("/"),
        base_model=raw["base_model"], production_adapter_uri=raw["production_adapter_uri"],
        replay_manifest=raw["replay_manifest"], old_golden_manifest=raw["old_golden_manifest"],
        new_holdout_ratio=float(raw["new_holdout_ratio"]), replay_ratio=float(raw["replay_ratio"]),
        min_samples=int(raw["min_samples"]), seed=int(raw["seed"]),
        old_wer_regression_max_pp=float(gate["old_wer_regression_max_pp"]),
        new_wer_improvement_min_pp=float(gate["new_wer_improvement_min_pp"]),
    )
