from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from google.api_core.exceptions import PreconditionFailed

from .config import load_config
from .gcs import GCS
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/stt.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    migration = sub.add_parser("migrate-status")
    migration.add_argument("--dry-run", action="store_true")
    snapshot = sub.add_parser("snapshot")
    snapshot.add_argument("--min-samples", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    gcs = GCS(cfg.project_id, cfg.bucket)
    if args.command == "migrate-status":
        changed, skipped = migrate_status(gcs, cfg, args.dry_run)
        print(json.dumps({"dry_run": args.dry_run, "changed": changed, "skipped": skipped}))
        return
    try:
        run_id, count, created = create_snapshot(gcs, cfg, args.min_samples or cfg.min_samples)
    except ActiveRunExists as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"snapshot_id": run_id or None, "eligible_count": count, "created": created}))


if __name__ == "__main__":
    main()
