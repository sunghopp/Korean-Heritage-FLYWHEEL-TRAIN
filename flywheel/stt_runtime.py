from __future__ import annotations

import hashlib
from pathlib import Path

import librosa
import torch
from peft import PeftConfig, PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from .gcs import GCS, parse_gs_uri


def local_file_for_uri(gcs: GCS, uri: str, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uri).suffix or ".wav"
    destination = cache_dir / f"{hashlib.sha256(uri.encode()).hexdigest()}{suffix}"
    if not destination.exists():
        bucket, name = parse_gs_uri(uri)
        gcs.client.bucket(bucket).blob(name).download_to_filename(destination)
    return destination


def download_adapter(gcs: GCS, uri: str, destination: Path) -> Path:
    bucket_name, prefix = parse_gs_uri(uri)
    prefix = prefix.rstrip("/") + "/"
    destination.mkdir(parents=True, exist_ok=True)
    for blob in gcs.client.bucket(bucket_name).list_blobs(prefix=prefix):
        relative = blob.name.removeprefix(prefix)
        if relative:
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(target)
    return destination


def load_stt_model(gcs: GCS, base_model: str, adapter_uri: str, work_dir: Path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    adapter_dir = download_adapter(gcs, adapter_uri, work_dir / "adapter")
    adapter_config = PeftConfig.from_pretrained(adapter_dir)
    actual_base_model = adapter_config.base_model_name_or_path or base_model
    processor = WhisperProcessor.from_pretrained(actual_base_model, language="Korean", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(actual_base_model)
    model = PeftModel.from_pretrained(base, adapter_dir).to(device).eval()
    model.generation_config.language = "ko"
    model.generation_config.task = "transcribe"
    return processor, model, device


def transcribe(gcs: GCS, processor, model, device: str, audio_uri: str, start: float, end: float, cache_dir: Path) -> str:
    path = local_file_for_uri(gcs, audio_uri, cache_dir)
    kwargs = {"sr": 16000, "offset": start}
    if end > start:
        kwargs["duration"] = end - start
    audio, _ = librosa.load(path, **kwargs)
    features = processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    with torch.no_grad():
        generated = model.generate(features)
    return processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
