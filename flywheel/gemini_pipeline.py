from __future__ import annotations

import argparse
import json

from .gemini_config import load_gemini_config
from .gemini_flywheel import finalize_tuning, submit_and_wait, submit_tuning
from .gcs import GCS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_id")
    parser.add_argument("--mode", required=True, choices=["submit", "wait", "finalize"])
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-wait-seconds", type=int, default=14400)
    parser.add_argument("--config", default="configs/gemini.yaml")
    args = parser.parse_args()
    cfg = load_gemini_config(args.config)
    gcs = GCS(cfg.project_id, cfg.bucket)
    if args.mode == "submit":
        result = submit_tuning(gcs, cfg, args.snapshot_id)
    elif args.mode == "wait":
        result = submit_and_wait(
            gcs, cfg, args.snapshot_id,
            poll_seconds=args.poll_seconds,
            max_wait_seconds=args.max_wait_seconds,
        )
    else:
        result = finalize_tuning(gcs, cfg, args.snapshot_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
