from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .config import Config
from .gcs import GCS, gs_uri


@dataclass(frozen=True)
class STTExample:
    sample_id: str
    label_uri: str
    label_generation: int
    audio_uri: str
    start: float
    end: float
    transcript: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_approved_for_stt(label: dict[str, Any]) -> bool:
    """Use the API's existing approval decision; do not recalculate confidence."""
    return label.get("status") == "approved" or label.get("review_status") == "approved"


def is_promoted(label: dict[str, Any]) -> bool:
    return bool(label.get("training_status", {}).get("stt", {}).get("promoted", False))


def _audio_uri(label: dict[str, Any], label_name: str, cfg: Config) -> str:
    saved = label.get("audio_filepath")
    if isinstance(saved, str) and saved:
        return saved if saved.startswith("gs://") else gs_uri(cfg.bucket, saved)
    file_id = label.get("id") or label_name.rsplit("/", 1)[-1].removesuffix(".json")
    return gs_uri(cfg.bucket, f"{cfg.audio_prefix}/{file_id}.wav")


def extract_examples(label: dict[str, Any], label_name: str, generation: int, cfg: Config) -> list[STTExample]:
    """Accept both API one-record labels and original Malmoi utterance labels."""
    audio_uri = _audio_uri(label, label_name, cfg)
    if isinstance(label.get("utterance"), list):
        result = []
        for index, utterance in enumerate(label["utterance"]):
            transcript = (utterance.get("dialect_form") or utterance.get("form") or "").strip()
            start, end = float(utterance.get("start", 0)), float(utterance.get("end", 0))
            if transcript and end > start:
                result.append(STTExample(
                    sample_id=str(utterance.get("id") or f"{label.get('id', label_name)}.{index}"),
                    label_uri=gs_uri(cfg.bucket, label_name), label_generation=generation,
                    audio_uri=audio_uri, start=start, end=end, transcript=transcript,
                ))
        return result

    transcript = (label.get("dialect_form") or label.get("form") or "").strip()
    # API labels represent one already-cropped request audio; start/end are not needed.
    if not transcript:
        return []
    return [STTExample(
        sample_id=str(label.get("id") or label_name.rsplit("/", 1)[-1].removesuffix(".json")),
        label_uri=gs_uri(cfg.bucket, label_name), label_generation=generation,
        audio_uri=audio_uri, start=0.0, end=-1.0, transcript=transcript,
    )]


def eligible_examples(gcs: GCS, cfg: Config) -> list[STTExample]:
    examples: list[STTExample] = []
    for blob in gcs.bucket.list_blobs(prefix=f"{cfg.text_prefix}/"):
        if not blob.name.endswith(".json"):
            continue
        label, generation = gcs.read_json(blob.name)
        if is_approved_for_stt(label) and not is_promoted(label):
            examples.extend(extract_examples(label, blob.name, generation, cfg))
    return examples


def snapshot_id(examples: Iterable[STTExample]) -> str:
    canonical = "\n".join(sorted(
        f"{item.sample_id}|{item.label_uri}|{item.label_generation}" for item in examples
    ))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def stable_holdout(examples: list[STTExample], ratio: float, seed: int) -> tuple[list[STTExample], list[STTExample]]:
    train, holdout = [], []
    for item in examples:
        digest = hashlib.sha256(f"{seed}:{item.sample_id}".encode()).digest()
        (holdout if int.from_bytes(digest[:8], "big") / 2**64 < ratio else train).append(item)
    return train, holdout
