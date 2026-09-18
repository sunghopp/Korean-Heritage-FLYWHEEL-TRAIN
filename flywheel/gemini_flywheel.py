"""GCS-only data flywheel for Jeju dialect -> standard Korean tuning.

The immutable snapshot and run objects are the durable state store; Firestore is
intentionally not used.  This module never stores or learns an ARS answer.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import sacrebleu
from google.api_core.exceptions import PreconditionFailed

from .gemini_config import GeminiConfig
from .gcs import GCS, gs_uri, parse_gs_uri

SYSTEM_INSTRUCTION = "당신은 제주 방언을 표준어로 정확하게 번역하는 전문 번역가입니다."


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status(record: dict[str, Any]) -> str:
    return str(record.get("status") or record.get("review_status") or "").lower()


def eligible_examples(gcs: GCS, cfg: GeminiConfig) -> list[dict[str, Any]]:
    """Return only human-approved, unpromoted translation pairs.

    Auto-approved Gemini output is useful operationally, but must not be used as
    its own ground truth for the tuning gate.  Human review makes the new-set
    BLEU comparison meaningful.
    """
    result: list[dict[str, Any]] = []
    for blob in gcs.bucket.list_blobs(prefix=f"{cfg.text_prefix}/"):
        if not blob.name.endswith(".json"):
            continue
        record, generation = gcs.read_json(blob.name)
        dialect = str(record.get("dialect_form") or record.get("form") or "").strip()
        standard = str(record.get("standard_form") or "").strip()
        gemini = (record.get("training_status") or {}).get("gemini") or {}
        is_human = str(record.get("reviewed_by") or "").lower() == "human"
        if (_status(record) != "approved" or gemini.get("promoted") is True
                or not dialect or not standard or (cfg.require_human_review and not is_human)):
            continue
        result.append({
            "sample_id": str(record.get("id") or blob.name.rsplit("/", 1)[-1].removesuffix(".json")),
            "source_json": gs_uri(cfg.bucket, blob.name), "source_generation": generation,
            "dialect_form": dialect, "standard_form": standard,
        })
    return sorted(result, key=lambda item: item["sample_id"])


def _id(examples: list[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{x['source_json']}:{x['source_generation']}:{x['dialect_form']}:{x['standard_form']}"
        for x in examples
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def create_snapshot(gcs: GCS, cfg: GeminiConfig) -> tuple[str | None, int, bool]:
    lock_name = f"{cfg.flywheel_prefix}/locks/active-snapshot.lock"
    try:
        gcs.write_json(lock_name, {"kind": "gemini", "started_at": _now()}, create_only=True)
    except PreconditionFailed as exc:
        raise RuntimeError("Another Gemini snapshot creation is active") from exc
    try:
        examples = eligible_examples(gcs, cfg)
        if len(examples) < cfg.min_samples:
            return None, len(examples), False
        snapshot_id = _id(examples)
        manifest_name = f"{cfg.flywheel_prefix}/snapshots/{snapshot_id}/manifest.jsonl"
        run_name = f"{cfg.flywheel_prefix}/runs/{snapshot_id}/run.json"
        try:
            gcs.write_text(manifest_name, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in examples), create_only=True)
            gcs.write_json(run_name, {
                "snapshot_id": snapshot_id, "status": "created", "example_count": len(examples),
                "manifest": gs_uri(cfg.bucket, manifest_name), "created_at": _now(),
            }, create_only=True)
        except PreconditionFailed:
            return snapshot_id, len(examples), False
        return snapshot_id, len(examples), True
    finally:
        gcs.bucket.blob(lock_name).delete()


def _read_jsonl(gcs: GCS, uri: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in gcs.download_uri(uri).splitlines() if line.strip()]


def _rank(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda x: hashlib.sha256(f"{seed}:{x['sample_id']}".encode()).hexdigest())


def _tuning_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "systemInstruction": {"role": "system", "parts": [{"text": SYSTEM_INSTRUCTION}]},
        "contents": [
            {"role": "user", "parts": [{"text": row["dialect_form"]}]},
            {"role": "model", "parts": [{"text": row["standard_form"]}]},
        ],
    }


def _write_dataset(gcs: GCS, name: str, rows: list[dict[str, Any]]) -> str:
    gcs.write_text(name, "".join(json.dumps(_tuning_row(row), ensure_ascii=False) + "\n" for row in rows), create_only=True)
    return gs_uri(gcs.bucket.name, name)


def prepare_tuning_data(gcs: GCS, cfg: GeminiConfig, snapshot_id: str) -> tuple[dict[str, Any], int]:
    run_name = f"{cfg.flywheel_prefix}/runs/{snapshot_id}/run.json"
    run, generation = gcs.read_json(run_name)
    if run["status"] not in {"created", "prepared"}:
        return run, generation
    new_rows = _read_jsonl(gcs, run["manifest"])
    ranked = _rank(new_rows, cfg.seed)
    validation_count = max(cfg.min_validation_samples, round(len(ranked) * cfg.validation_ratio))
    if validation_count >= len(ranked):
        raise ValueError("Not enough snapshot rows after validation split; increase min_samples")
    validation = ranked[:validation_count]
    new_train = ranked[validation_count:]
    replay = _read_jsonl(gcs, cfg.replay_manifest)
    replay_limit = min(len(replay), round(len(new_train) * cfg.replay_ratio))
    train = new_train + _rank(replay, cfg.seed)[:replay_limit]
    prefix = f"{cfg.flywheel_prefix}/snapshots/{snapshot_id}"
    train_uri = _write_dataset(gcs, f"{prefix}/train.jsonl", train)
    validation_uri = _write_dataset(gcs, f"{prefix}/validation.jsonl", validation)
    run.update({
        "status": "prepared", "new_train_count": len(new_train), "replay_train_count": replay_limit,
        "validation_count": len(validation), "train_dataset": train_uri, "validation_dataset": validation_uri,
        "prepared_at": _now(),
    })
    gcs.write_json(run_name, run, generation=generation)
    return run, generation + 1


def _client(cfg: GeminiConfig):
    os.environ["GOOGLE_GENAI_USE_ENTERPRISE"] = "True"
    os.environ["GOOGLE_CLOUD_PROJECT"] = cfg.project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = "us-central1"
    from google import genai
    from google.genai.types import HttpOptions
    return genai.Client(http_options=HttpOptions(api_version="v1beta1"))


def submit_tuning(gcs: GCS, cfg: GeminiConfig, snapshot_id: str) -> dict[str, Any]:
    run, _ = prepare_tuning_data(gcs, cfg, snapshot_id)
    if run["status"] == "submitted":
        return run
    if run["status"] != "prepared":
        raise RuntimeError(f"Snapshot {snapshot_id} cannot be submitted from status={run['status']}")
    from google.genai.types import CreateTuningJobConfig, TuningDataset
    client = _client(cfg)
    job = client.tunings.tune(
        base_model=cfg.base_model,
        training_dataset=TuningDataset(gcs_uri=run["train_dataset"]),
        config=CreateTuningJobConfig(
            tuned_model_display_name=f"jeju-translation-{snapshot_id}",
            validation_dataset=TuningDataset(gcs_uri=run["validation_dataset"]),
        ),
    )
    name = f"{cfg.flywheel_prefix}/runs/{snapshot_id}/run.json"
    latest, generation = gcs.read_json(name)
    if latest["status"] == "submitted":
        return latest
    latest.update({"status": "submitted", "tuning_job": job.name, "submitted_at": _now()})
    gcs.write_json(name, latest, generation=generation)
    return latest


def _state_name(job: Any) -> str:
    state = getattr(job, "state", "")
    return getattr(state, "name", str(state)).upper()


def _translate(client: Any, endpoint: str, text: str) -> str:
    response = client.models.generate_content(
        model=endpoint,
        contents=text,
        config={"system_instruction": SYSTEM_INSTRUCTION, "temperature": 0},
    )
    return (response.text or "").strip()


def _bleu(client: Any, endpoint: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    predictions = [_translate(client, endpoint, row["dialect_form"]) for row in rows]
    references = [row["standard_form"] for row in rows]
    bleu = sacrebleu.corpus_bleu(predictions, [references], tokenize="none").score
    exact = sum(a == b for a, b in zip(predictions, references)) / len(rows) * 100
    return {"bleu": round(bleu, 4), "exact_match_rate": round(exact, 4), "count": len(rows)}


def finalize_tuning(gcs: GCS, cfg: GeminiConfig, snapshot_id: str) -> dict[str, Any]:
    name = f"{cfg.flywheel_prefix}/runs/{snapshot_id}/run.json"
    run, generation = gcs.read_json(name)
    if run["status"] == "approved":
        return run
    if run["status"] != "submitted":
        raise RuntimeError(f"Snapshot {snapshot_id} cannot finalize from status={run['status']}")
    client = _client(cfg)
    job = client.tunings.get(name=run["tuning_job"])
    state = _state_name(job)
    if state not in {"SUCCEEDED", "SUCCESS"}:
        run.update({"last_observed_tuning_state": state, "last_checked_at": _now()})
        gcs.write_json(name, run, generation=generation)
        return run
    candidate_endpoint = str(job.tuned_model.endpoint)
    validation = _read_jsonl(gcs, run["manifest"])
    validation_ids = {row["sample_id"] for row in _rank(validation, cfg.seed)[:run["validation_count"]]}
    new_validation = [row for row in validation if row["sample_id"] in validation_ids]
    golden = _read_jsonl(gcs, cfg.old_golden_manifest)
    baseline_old = _bleu(client, cfg.baseline_endpoint, golden)
    candidate_old = _bleu(client, candidate_endpoint, golden)
    baseline_new = _bleu(client, cfg.baseline_endpoint, new_validation)
    candidate_new = _bleu(client, candidate_endpoint, new_validation)
    old_delta = candidate_old["bleu"] - baseline_old["bleu"]
    new_delta = candidate_new["bleu"] - baseline_new["bleu"]
    approved = old_delta >= -cfg.old_bleu_regression_max and new_delta >= cfg.new_bleu_improvement_min
    report = {
        "snapshot_id": snapshot_id, "evaluated_at": _now(), "baseline_endpoint": cfg.baseline_endpoint,
        "candidate_endpoint": candidate_endpoint, "baseline_old": baseline_old, "candidate_old": candidate_old,
        "baseline_new": baseline_new, "candidate_new": candidate_new,
        "old_bleu_delta": round(old_delta, 4), "new_bleu_delta": round(new_delta, 4),
        "approved": approved,
        "gate": {"old_bleu_regression_max": cfg.old_bleu_regression_max, "new_bleu_improvement_min": cfg.new_bleu_improvement_min},
    }
    gcs.write_json(f"{cfg.flywheel_prefix}/evaluations/{snapshot_id}.json", report, create_only=True)
    run.update({"status": "approved" if approved else "rejected", "candidate_endpoint": candidate_endpoint,
                "evaluated_at": _now(), "approved": approved})
    gcs.write_json(name, run, generation=generation)
    if approved:
        release = {**report, "tuning_job": run["tuning_job"], "release_created_at": _now(),
                   "activation": "manual_api_integration_required"}
        gcs.write_json(f"{cfg.flywheel_prefix}/releases/gemini-{snapshot_id}.json", release, create_only=True)
        gcs.write_json(f"{cfg.flywheel_prefix}/releases/current.json", release)
        _mark_promoted(gcs, snapshot_id, _read_jsonl(gcs, run["manifest"]))
    return run


def _mark_promoted(gcs: GCS, snapshot_id: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        bucket, name = parse_gs_uri(row["source_json"])
        if bucket != gcs.bucket.name:
            raise ValueError("Cross-bucket source JSON is not supported")
        record, generation = gcs.read_json(name)
        status = record.setdefault("training_status", {}).setdefault("gemini", {})
        status.update({"promoted": True, "last_snapshot_id": snapshot_id, "last_trained_at": _now()})
        try:
            gcs.write_json(name, record, generation=generation)
        except PreconditionFailed:
            # A reviewer changed the JSON after its immutable snapshot.  Never
            # overwrite that human change; it remains eligible for a later run.
            continue


def bootstrap_baseline(gcs: GCS, cfg: GeminiConfig, source_uri: str, golden_count: int, replay_count: int) -> dict[str, int]:
    raw = _read_jsonl(gcs, source_uri)
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(raw):
        contents = item.get("contents") or []
        if len(contents) < 2:
            continue
        dialect = str(contents[0].get("parts", [{}])[0].get("text") or "").strip()
        standard = str(contents[1].get("parts", [{}])[0].get("text") or "").strip()
        if dialect and standard:
            rows.append({"sample_id": f"original-{index:06d}", "dialect_form": dialect, "standard_form": standard})
    ranked = _rank(rows, cfg.seed)
    if len(ranked) < golden_count + replay_count:
        raise ValueError(f"Need {golden_count + replay_count} valid original translation pairs, found {len(ranked)}")
    golden_name = cfg.old_golden_manifest.removeprefix(f"gs://{cfg.bucket}/")
    replay_name = cfg.replay_manifest.removeprefix(f"gs://{cfg.bucket}/")
    gcs.write_text(golden_name, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ranked[:golden_count]), create_only=True)
    gcs.write_text(replay_name, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in ranked[golden_count:golden_count + replay_count]), create_only=True)
    return {"golden_count": golden_count, "replay_count": replay_count}
