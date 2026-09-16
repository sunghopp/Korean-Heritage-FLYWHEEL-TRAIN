# Korean Heritage STT Flywheel

This repository contains the first stage of the data flywheel: incremental
Whisper Jeju LoRA training.  It never changes the existing dataset paths or
label schema.  The only added label field is `training_status`.

## Commands

```bash
python -m flywheel.cli migrate-status --dry-run
python -m flywheel.cli migrate-status
python -m flywheel.cli snapshot --min-samples 100
```

`snapshot` selects `status=approved` labels whose
`training_status.stt.promoted` is false.  It creates an immutable manifest and
a GCS run record with create-only preconditions, so the same snapshot cannot
be started twice.  The Cloud Run training job receives the snapshot id, builds
its train/new-holdout split, mixes in a fixed replay manifest, and writes a
candidate adapter and evaluation report.  Only `promote` changes the source
labels to `promoted=true`.

Promotion first copies the current production adapter to
`whisper-model-weights/backups/{release_id}/`. Only after that backup succeeds
does it copy the candidate to the stable production prefix
`whisper-model-weights/whisper-jeju-lora-final/`. `releases/current.json`
records the candidate and backup locations for audit and rollback.

Before the first training run, create these immutable manifests once:

* `flywheel/stt/replay-v1.jsonl`: approved historical examples used for replay.
* `flywheel/stt/eval/old-golden-v1.jsonl`: historical examples never used for training.

Both use the same JSONL record format as a snapshot manifest.

## Required GCP setup

The runtime service account needs read/write access to the dataset bucket and
read access to the production adapter prefix. The GitHub Actions service account
needs permission to deploy/execute the Cloud Run Job and restart `jeju-backend`
after a successful promotion. Configure these repository secrets:

* `GCP_PROJECT_ID`, `GCP_WORKLOAD_IDENTITY_PROVIDER`
* `GCP_FLYWHEEL_SERVICE_ACCOUNT` (GitHub Actions identity)
* `GCP_STT_RUNTIME_SERVICE_ACCOUNT` (Cloud Run Job identity)
