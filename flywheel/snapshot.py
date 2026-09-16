from __future__ import annotations

from datetime import datetime, timezone

from google.api_core.exceptions import PreconditionFailed

from .config import Config
from .dataset import eligible_examples, snapshot_id
from .gcs import GCS


class ActiveRunExists(RuntimeError):
    pass


def create_snapshot(gcs: GCS, cfg: Config, min_samples: int) -> tuple[str, int, bool]:
    lock_name = f"{cfg.flywheel_prefix}/locks/active-run.lock"
    lock = {"started_at": datetime.now(timezone.utc).isoformat(), "kind": "stt"}
    try:
        gcs.write_json(lock_name, lock, create_only=True)
    except PreconditionFailed as exc:
        raise ActiveRunExists("Another STT flywheel run holds active-run.lock") from exc

    try:
        examples = eligible_examples(gcs, cfg)
        if len(examples) < min_samples:
            return "", len(examples), False
        run_id = snapshot_id(examples)
        manifest_name = f"{cfg.flywheel_prefix}/snapshots/{run_id}/manifest.jsonl"
        run_name = f"{cfg.flywheel_prefix}/runs/{run_id}/run.json"
        manifest = "\n".join(__import__("json").dumps(x.as_dict(), ensure_ascii=False) for x in examples) + "\n"
        try:
            gcs.write_text(manifest_name, manifest, create_only=True)
            gcs.write_json(run_name, {
                "snapshot_id": run_id, "status": "created", "example_count": len(examples),
                "manifest": f"gs://{cfg.bucket}/{manifest_name}", "created_at": datetime.now(timezone.utc).isoformat(),
            }, create_only=True)
        except PreconditionFailed:
            # Immutable manifest/run already proves this exact dataset was submitted.
            return run_id, len(examples), False
        return run_id, len(examples), True
    finally:
        # The lock only serializes snapshot creation; the run.json object is the
        # durable idempotency record for the later Cloud Run jobs.
        gcs.bucket.blob(lock_name).delete()
