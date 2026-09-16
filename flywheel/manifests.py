from __future__ import annotations

import json
from pathlib import Path

from .dataset import STTExample
from .gcs import GCS


def read_manifest(gcs: GCS, uri: str) -> list[STTExample]:
    return [STTExample(**row) for row in (
        json.loads(line) for line in gcs.download_uri(uri).splitlines() if line.strip()
    )]


def write_manifest(path: Path, examples: list[STTExample]) -> None:
    path.write_text("".join(json.dumps(x.as_dict(), ensure_ascii=False) + "\n" for x in examples), encoding="utf-8")
