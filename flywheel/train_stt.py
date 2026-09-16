from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import torch
from datasets import Dataset
from peft import PeftConfig, PeftModel
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, WhisperProcessor

from .config import Config
from .dataset import STTExample, stable_holdout
from .gcs import GCS, parse_gs_uri
from .manifests import read_manifest, write_manifest
from .stt_runtime import download_adapter, local_file_for_uri


@dataclass
class Collator:
    processor: Any
    gcs: GCS
    cache_dir: Path

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        inputs = []
        for item in features:
            path = local_file_for_uri(self.gcs, item["audio_uri"], self.cache_dir)
            kwargs = {"sr": 16000, "offset": item["start"]}
            if item["end"] > item["start"]:
                kwargs["duration"] = item["end"] - item["start"]
            audio, _ = librosa.load(path, **kwargs)
            inputs.append({"input_features": self.processor.feature_extractor(audio, sampling_rate=16000).input_features[0]})
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")
        labels = self.processor.tokenizer.pad([{"input_ids": item["labels"]} for item in features], return_tensors="pt")
        batch["labels"] = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return batch


def _dataset(examples: list[STTExample], processor: WhisperProcessor) -> Dataset:
    rows = []
    for item in examples:
        row = item.as_dict()
        row["labels"] = processor.tokenizer(item.transcript).input_ids
        rows.append(row)
    return Dataset.from_list(rows)


def train_candidate(gcs: GCS, cfg: Config, snapshot_uri: str, output_dir: Path) -> tuple[Path, str]:
    new_examples = read_manifest(gcs, snapshot_uri)
    train_new, holdout_new = stable_holdout(new_examples, cfg.new_holdout_ratio, cfg.seed)
    replay = read_manifest(gcs, cfg.replay_manifest)
    replay_count = min(len(replay), round(len(train_new) * cfg.replay_ratio))
    train_examples = train_new + replay[:replay_count]
    if not train_new or not holdout_new:
        raise ValueError("Snapshot is too small to produce both train and new-holdout splits")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir / "new_holdout.jsonl", holdout_new)
    adapter_dir = download_adapter(gcs, cfg.production_adapter_uri, output_dir / "production_adapter")
    adapter_config = PeftConfig.from_pretrained(adapter_dir)
    actual_base_model = adapter_config.base_model_name_or_path or cfg.base_model
    processor = WhisperProcessor.from_pretrained(actual_base_model, language="Korean", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(actual_base_model)
    model = PeftModel.from_pretrained(base, adapter_dir, is_trainable=True)
    model.generation_config.language = "ko"
    model.generation_config.task = "transcribe"

    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"), per_device_train_batch_size=4,
        gradient_accumulation_steps=4, learning_rate=1e-4, num_train_epochs=2,
        fp16=torch.cuda.is_available(), logging_steps=10, save_strategy="epoch",
        dataloader_num_workers=2, dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False, report_to=[], seed=cfg.seed,
    )
    trainer = Seq2SeqTrainer(
        args=args, model=model, train_dataset=_dataset(train_examples, processor),
        data_collator=Collator(processor, gcs, output_dir / "audio_cache"), processing_class=processor,
    )
    trainer.train()
    adapter_output = output_dir / "adapter"
    trainer.model.save_pretrained(adapter_output)
    processor.save_pretrained(adapter_output)
    return adapter_output, str(output_dir / "new_holdout.jsonl")
