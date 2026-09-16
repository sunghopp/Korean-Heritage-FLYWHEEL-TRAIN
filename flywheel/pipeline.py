from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .config import load_config
from .evaluate_stt import evaluate_candidate
from .gcs import GCS, gs_uri
from .promote_stt import promote
from .train_stt import train_candidate


def upload_tree(gcs: GCS, local: Path, destination_uri: str) -> None:
    bucket, prefix = destination_uri[5:].split("/", 1)
    for file in local.rglob("*"):
        if file.is_file():
            gcs.client.bucket(bucket).blob(f"{prefix.rstrip('/')}/{file.relative_to(local).as_posix()}").upload_from_filename(file)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_id")
    parser.add_argument("--config", default="configs/stt.yaml")
    parser.add_argument("--work-dir", default="/tmp/flywheel")
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    gcs = GCS(cfg.project_id, cfg.bucket)
    # After the first promotion, continue from the latest accepted adapter,
    # not from the adapter hard-coded in the initial configuration.
    try:
        current = json.loads(gcs.download_uri(gs_uri(cfg.bucket, f"{cfg.flywheel_prefix}/releases/current.json")))
        if isinstance(current.get("stt_adapter_uri"), str):
            cfg = replace(cfg, production_adapter_uri=current["stt_adapter_uri"])
    except Exception:
        pass  # First run: use production_adapter_uri from stt.yaml.
    snapshot_uri = gs_uri(cfg.bucket, f"{cfg.flywheel_prefix}/snapshots/{args.snapshot_id}/manifest.jsonl")
    work_dir = Path(args.work_dir) / args.snapshot_id
    adapter_dir, holdout_path = train_candidate(gcs, cfg, snapshot_uri, work_dir)
    candidate_uri = gs_uri(cfg.bucket, f"{cfg.flywheel_prefix}/candidates/{args.snapshot_id}/adapter")
    upload_tree(gcs, adapter_dir, candidate_uri)
    holdout_uri = gs_uri(cfg.bucket, f"{cfg.flywheel_prefix}/snapshots/{args.snapshot_id}/new_holdout.jsonl")
    gcs.client.bucket(cfg.bucket).blob(holdout_uri[5:].split("/", 1)[1]).upload_from_filename(holdout_path)
    report_uri = gs_uri(cfg.bucket, f"{cfg.flywheel_prefix}/evaluations/{args.snapshot_id}.json")
    report = evaluate_candidate(gcs, cfg, candidate_uri, holdout_uri, report_uri, work_dir / "eval")
    run_name = f"{cfg.flywheel_prefix}/runs/{args.snapshot_id}/run.json"
    run, generation = gcs.read_json(run_name)
    run.update({"status": "evaluated", "candidate_adapter_uri": candidate_uri, "evaluation_uri": report_uri, "approved": report["approved"]})
    gcs.write_json(run_name, run, generation=generation)
    if args.promote and report["approved"]:
        promote(gcs, cfg, args.snapshot_id, candidate_uri, report_uri)
        run, generation = gcs.read_json(run_name)
        run["status"] = "promoted"
        gcs.write_json(run_name, run, generation=generation)
    elif not report["approved"]:
        run, generation = gcs.read_json(run_name)
        run["status"] = "rejected"
        gcs.write_json(run_name, run, generation=generation)
    print({"approved": report["approved"], "candidate_uri": candidate_uri, "report_uri": report_uri})


if __name__ == "__main__":
    main()
