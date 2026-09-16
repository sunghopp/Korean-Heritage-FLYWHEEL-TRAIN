from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import Config
from .gcs import GCS, parse_gs_uri
from .manifests import read_manifest


def promote(gcs: GCS, cfg: Config, snapshot_id: str, candidate_uri: str, report_uri: str) -> None:
    report = json.loads(gcs.download_uri(report_uri))
    if not report.get("approved"):
        raise ValueError("Refusing promotion: evaluation gate did not pass")
    release_id = f"stt-{snapshot_id}"
    backup_uri = f"gs://{cfg.bucket}/whisper-model-weights/backups/{release_id}"
    production_uri = f"gs://{cfg.bucket}/whisper-model-weights/whisper-jeju-lora-final"
    release_name = f"{cfg.flywheel_prefix}/releases/{release_id}.json"
    current_name = f"{cfg.flywheel_prefix}/releases/current.json"
    previous = None
    try:
        previous, current_generation = gcs.read_json(current_name)
    except Exception:
        current_generation = None
    # Preserve a complete immutable copy before changing the stable production
    # prefix.  The running API has the old adapter in memory, so it is not
    # exposed to a partially copied prefix; it is restarted only afterwards.
    backup_files = gcs.copy_prefix(cfg.production_adapter_uri, backup_uri, overwrite=False)
    candidate_files = gcs.copy_prefix(candidate_uri, production_uri, overwrite=True)
    gcs.delete_prefix_except(production_uri, set(candidate_files))
    release = {
        "release_id": release_id, "previous_release": previous.get("release_id") if previous else None,
        "stt_adapter_uri": production_uri, "candidate_adapter_uri": candidate_uri,
        "backup_adapter_uri": backup_uri, "backup_files": backup_files,
        "snapshot_id": snapshot_id, "evaluation_uri": report_uri,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
    }
    gcs.write_json(release_name, release, create_only=True)
    gcs.write_json(current_name, release, generation=current_generation)

    # Source labels are only marked after the immutable release record exists.
    manifest_uri = f"gs://{cfg.bucket}/{cfg.flywheel_prefix}/snapshots/{snapshot_id}/manifest.jsonl"
    seen: set[tuple[str, int]] = set()
    for example in read_manifest(gcs, manifest_uri):
        name = parse_gs_uri(example.label_uri)[1]
        key = (name, example.label_generation)
        if key in seen:
            continue
        seen.add(key)
        label, generation = gcs.read_json(name)
        state = label.setdefault("training_status", {})
        stt = state.setdefault("stt", {})
        stt.update({"promoted": True, "last_snapshot_id": snapshot_id, "last_trained_at": release["promoted_at"]})
        # Use the latest generation: human label fixes made during training are preserved.
        gcs.write_json(name, label, generation=generation)
