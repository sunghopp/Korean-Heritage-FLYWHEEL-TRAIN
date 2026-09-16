from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone

from google.api_core.exceptions import PreconditionFailed

from .config import Config, load_config
from .dataset import STTExample, extract_examples
from .gcs import GCS, parse_gs_uri
from .snapshot import ActiveRunExists, create_snapshot


def migrate_status(gcs: GCS, cfg, dry_run: bool) -> tuple[int, int]:
    changed = skipped = 0
    for blob in gcs.bucket.list_blobs(prefix=f"{cfg.text_prefix}/"):
        if not blob.name.endswith(".json"):
            continue
        record, generation = gcs.read_json(blob.name)
        if "training_status" in record:
            skipped += 1
            continue
        record["training_status"] = {
            "stt": {"promoted": False, "last_snapshot_id": None, "last_trained_at": None},
            "gemini": {"promoted": False, "last_snapshot_id": None, "last_trained_at": None},
        }
        if not dry_run:
            try:
                gcs.write_json(blob.name, record, generation=generation)
            except PreconditionFailed:
                # A reviewer/API changed this record while migration was running.
                skipped += 1
                continue
        changed += 1
    return changed, skipped


def _manifest_text(examples: list[STTExample]) -> str:
    return "".join(json.dumps(example.as_dict(), ensure_ascii=False) + "\n" for example in examples)


def _stable_sample(examples: list[STTExample], seed: int, limit: int) -> list[STTExample]:
    return sorted(
        examples,
        key=lambda item: hashlib.sha256(f"{seed}:{item.sample_id}".encode()).hexdigest(),
    )[:limit]


def bootstrap_baseline(
    gcs: GCS,
    cfg: Config,
    golden_files: int,
    replay_files: int,
    golden_utterances: int,
    replay_utterances: int,
) -> dict[str, int]:
    """Create disjoint, immutable Golden and replay manifests from original labels.

    Selection happens at recording-file level, so utterances from the same audio
    file cannot appear in both sets. Original labels are never modified.
    """
    if golden_files < 1 or replay_files < 1:
        raise ValueError("golden-files and replay-files must both be positive")
    baseline_cfg = replace(
        cfg,
        text_prefix=cfg.baseline_text_prefix,
        audio_prefix=cfg.baseline_audio_prefix,
    )
    labels = [
        blob for blob in gcs.bucket.list_blobs(prefix=f"{cfg.baseline_text_prefix}/")
        if blob.name.endswith(".json")
    ]
    available_audio_objects = {
        blob.name
        for blob in gcs.bucket.list_blobs(prefix=f"{cfg.baseline_audio_prefix}/")
        if not blob.name.endswith("/")
    }
    valid_recordings: list[tuple[str, list[STTExample]]] = []
    for blob in labels:
        label, generation = gcs.read_json(blob.name)
        examples = extract_examples(label, blob.name, generation, baseline_cfg)
        if not examples:
            continue
        audio_bucket, audio_name = parse_gs_uri(examples[0].audio_uri)
        # A label can be used only when its exact source audio object is present.
        # This prevents a historical label-only file from failing Cloud Run training.
        if audio_bucket == cfg.bucket and audio_name in available_audio_objects:
            valid_recordings.append((blob.name, examples))

    ranked = sorted(
        valid_recordings,
        key=lambda recording: hashlib.sha256(f"{cfg.seed}:{recording[0]}".encode()).hexdigest(),
    )
    required_recordings = golden_files + replay_files
    if len(ranked) < required_recordings:
        raise ValueError(
            f"Need {required_recordings} original labels with matching audio, found {len(ranked)}"
        )

    def examples_for(recordings: list[tuple[str, list[STTExample]]]) -> list[STTExample]:
        return [example for _, examples in recordings for example in examples]

    golden = _stable_sample(examples_for(ranked[:golden_files]), cfg.seed, golden_utterances)
    replay = _stable_sample(
        examples_for(ranked[golden_files:golden_files + replay_files]),
        cfg.seed,
        replay_utterances,
    )
    if not golden or not replay:
        raise ValueError("Baseline selection produced an empty manifest")
    golden_name = cfg.old_golden_manifest.removeprefix(f"gs://{cfg.bucket}/")
    replay_name = cfg.replay_manifest.removeprefix(f"gs://{cfg.bucket}/")
    gcs.write_text(golden_name, _manifest_text(golden), create_only=True)
    try:
        gcs.write_text(replay_name, _manifest_text(replay), create_only=True)
    except Exception:
        # Golden is intentionally immutable. Do not silently replace it if the
        # second write fails; report the exact object that needs operator review.
        raise RuntimeError(f"Golden manifest created but replay manifest failed: gs://{cfg.bucket}/{golden_name}")
    return {
        "golden_recordings": golden_files,
        "golden_utterances": len(golden),
        "replay_recordings": replay_files,
        "replay_utterances": len(replay),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stt.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    migration = sub.add_parser("migrate-status")
    migration.add_argument("--dry-run", action="store_true")
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--min-samples", type=int)
    baseline = sub.add_parser("bootstrap-baseline")
    baseline.add_argument("--golden-files", type=int, default=15)
    baseline.add_argument("--replay-files", type=int, default=60)
    baseline.add_argument("--golden-utterances", type=int, default=300)
    baseline.add_argument("--replay-utterances", type=int, default=3000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    gcs = GCS(cfg.project_id, cfg.bucket)
    if args.command == "migrate-status":
        changed, skipped = migrate_status(gcs, cfg, args.dry_run)
        print(json.dumps({"dry_run": args.dry_run, "changed": changed, "skipped": skipped}))
        return
    if args.command == "bootstrap-baseline":
        print(json.dumps(bootstrap_baseline(
            gcs,
            cfg,
            args.golden_files,
            args.replay_files,
            args.golden_utterances,
            args.replay_utterances,
        )))
        return
    try:
        run_id, count, created = create_snapshot(gcs, cfg, args.min_samples or cfg.min_samples)
    except ActiveRunExists as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"snapshot_id": run_id or None, "eligible_count": count, "created": created}))


if __name__ == "__main__":
    main()
