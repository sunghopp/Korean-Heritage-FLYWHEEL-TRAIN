from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GeminiConfig:
    project_id: str
    bucket: str
    text_prefix: str
    flywheel_prefix: str
    baseline_endpoint: str
    base_model: str
    replay_manifest: str
    old_golden_manifest: str
    min_samples: int
    validation_ratio: float
    min_validation_samples: int
    replay_ratio: float
    seed: int
    require_human_review: bool
    old_bleu_regression_max: float
    new_bleu_improvement_min: float


def load_gemini_config(path: str) -> GeminiConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    gate = raw["promotion"]
    return GeminiConfig(
        project_id=str(raw["project_id"]), bucket=raw["bucket"],
        text_prefix=raw["text_prefix"].strip("/"),
        flywheel_prefix=raw["flywheel_prefix"].strip("/"),
        baseline_endpoint=raw["baseline_endpoint"], base_model=raw["base_model"],
        replay_manifest=raw["replay_manifest"], old_golden_manifest=raw["old_golden_manifest"],
        min_samples=int(raw["min_samples"]), validation_ratio=float(raw["validation_ratio"]),
        min_validation_samples=int(raw["min_validation_samples"]), replay_ratio=float(raw["replay_ratio"]),
        seed=int(raw["seed"]), require_human_review=bool(raw["require_human_review"]),
        old_bleu_regression_max=float(gate["old_bleu_regression_max"]),
        new_bleu_improvement_min=float(gate["new_bleu_improvement_min"]),
    )
