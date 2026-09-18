from __future__ import annotations

import argparse
import json

from .gemini_config import load_gemini_config
from .gemini_flywheel import finalize_tuning, submit_tuning
from .gcs import GCS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot_id")
    parser.add_argument("--mode", required=True, choices=["submit", "finalize"])
    parser.add_argument("--config", default="configs/gemini.yaml")
    args = parser.parse_args()
    cfg = load_gemini_config(args.config)
    gcs = GCS(cfg.project_id, cfg.bucket)
    result = submit_tuning(gcs, cfg, args.snapshot_id) if args.mode == "submit" else finalize_tuning(gcs, cfg, args.snapshot_id)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
