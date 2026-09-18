from __future__ import annotations

import argparse
import json

from .gemini_config import load_gemini_config
from .gemini_flywheel import bootstrap_baseline, create_snapshot
from .gcs import GCS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gemini.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("snapshot")
    boot = sub.add_parser("bootstrap-baseline")
    boot.add_argument("--source-uri", required=True)
    boot.add_argument("--golden-count", type=int, default=300)
    boot.add_argument("--replay-count", type=int, default=3000)
    args = parser.parse_args()
    cfg = load_gemini_config(args.config)
    gcs = GCS(cfg.project_id, cfg.bucket)
    if args.command == "snapshot":
        snapshot_id, count, created = create_snapshot(gcs, cfg)
        print(json.dumps({"snapshot_id": snapshot_id, "eligible_count": count, "created": created}))
    else:
        print(json.dumps(bootstrap_baseline(gcs, cfg, args.source_uri, args.golden_count, args.replay_count)))


if __name__ == "__main__":
    main()
