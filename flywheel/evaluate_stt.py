from __future__ import annotations

import json
from pathlib import Path

from jiwer import wer

from .config import Config
from .gcs import GCS
from .manifests import read_manifest
from .stt_runtime import load_stt_model, transcribe


def normalize(text: str) -> str:
    return " ".join(text.strip().split())


def score(gcs: GCS, cfg: Config, adapter_uri: str, manifest_uri: str, work_dir: Path) -> dict:
    examples = read_manifest(gcs, manifest_uri)
    processor, model, device = load_stt_model(gcs, cfg.base_model, adapter_uri, work_dir / "model")
    references, predictions = [], []
    for item in examples:
        references.append(normalize(item.transcript))
        predictions.append(normalize(transcribe(gcs, processor, model, device, item.audio_uri, item.start, item.end, work_dir / "audio")))
    value = wer(references, predictions) * 100 if references else 0.0
    return {"count": len(examples), "wer": round(value, 4), "predictions": [
        {"sample_id": item.sample_id, "reference": ref, "prediction": pred}
        for item, ref, pred in zip(examples, references, predictions)
    ]}


def evaluate_candidate(gcs: GCS, cfg: Config, candidate_uri: str, snapshot_uri: str, output_uri: str, work_dir: Path) -> dict:
    baseline_old = score(gcs, cfg, cfg.production_adapter_uri, cfg.old_golden_manifest, work_dir / "baseline-old")
    candidate_old = score(gcs, cfg, candidate_uri, cfg.old_golden_manifest, work_dir / "candidate-old")
    baseline_new = score(gcs, cfg, cfg.production_adapter_uri, snapshot_uri, work_dir / "baseline-new")
    candidate_new = score(gcs, cfg, candidate_uri, snapshot_uri, work_dir / "candidate-new")
    old_regression = candidate_old["wer"] - baseline_old["wer"]
    new_improvement = baseline_new["wer"] - candidate_new["wer"]
    approved = old_regression <= cfg.old_wer_regression_max_pp and new_improvement >= cfg.new_wer_improvement_min_pp
    report = {
        "approved": approved, "candidate_adapter_uri": candidate_uri,
        "baseline_old": baseline_old, "candidate_old": candidate_old,
        "baseline_new": baseline_new, "candidate_new": candidate_new,
        "old_wer_regression_pp": round(old_regression, 4), "new_wer_improvement_pp": round(new_improvement, 4),
        "gate": {"old_wer_regression_max_pp": cfg.old_wer_regression_max_pp, "new_wer_improvement_min_pp": cfg.new_wer_improvement_min_pp},
    }
    bucket, name = output_uri[5:].split("/", 1)
    gcs.client.bucket(bucket).blob(name).upload_from_string(json.dumps(report, ensure_ascii=False, indent=2), content_type="application/json")
    return report
