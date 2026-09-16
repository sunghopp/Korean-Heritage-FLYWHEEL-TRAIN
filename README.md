# Korean Heritage STT Flywheel

제주어 Whisper LoRA의 증분 학습을 위한 GCS 전용 Data Flywheel 프로젝트입니다. API 서버와 학습 Pipeline의 책임을 분리하며, Firestore는 사용하지 않습니다.

## 목적과 범위

- Web/API에서 수집·승인된 음성 및 JSON을 누적하고, 일정 수량이 되면 Cloud Run Job에서 증분 LoRA 학습을 수행합니다.
- 기존 사용자 입력 성능 저하를 막기 위해 기존 Golden Set과 신규 Holdout Set의 WER을 모두 비교한 뒤에만 가중치를 교체합니다.
- 전체 재학습 대신 현재 운영 LoRA에서 시작하고, 과거 Replay 데이터를 섞어 학습 비용과 망각을 줄입니다.
- 이 저장소는 STT 전용입니다. Gemini 제주어 번역 모델 조정과 RAG Scenario는 별도 단계에서 다룹니다.

## 저장소와 GCS 책임 분리

| 구분 | 책임 | 위치 |
| --- | --- | --- |
| API 서버 | 신규 음성·라벨 JSON 생성, 기존 Confidence 기반 승인 | `Korean-Heritage-API-SVR` |
| 학습 Pipeline | 상태 마이그레이션, 스냅샷, 학습, 평가, 승격 | 이 저장소 |
| 신규 수집 데이터 | API가 생성한 Audio·Text | `dataset/extracted/Audio`, `dataset/extracted/Text` |
| 기존 기준 데이터 | 변하지 않는 과거 Audio·Label | `extracted/extracted/Audio/Audio`, `extracted/extracted/Text/Text/Label` |
| 운영 LoRA | 현재 API가 읽는 어댑터 | `whisper-model-weights/whisper-jeju-lora-final/` |

기존 JSON의 발화·번역·Confidence 필드는 변경하지 않습니다. 신규 JSON에는 아래 상태만 추가합니다.

```json
"training_status": {
  "stt": {
    "promoted": false,
    "last_snapshot_id": null,
    "last_trained_at": null
  },
  "gemini": {
    "promoted": false,
    "last_snapshot_id": null,
    "last_trained_at": null
  }
}
```

`stt.promoted`가 `false`인 승인 데이터만 새 스냅샷 후보입니다. 승격이 성공한 뒤에만 해당 입력 JSON이 `true`로 바뀌므로, 평가 실패·학습 실패한 데이터는 다음 실행에서 다시 사용할 수 있습니다.

## 테스트 기준 세트 명세

`bootstrap-baseline`은 원본 녹음 파일 단위로 결정론적 해시 선택을 합니다. 선택 전 라벨이 가리키는 정확한 GCS Audio 객체가 실제로 존재하는지도 검증합니다. 따라서 같은 녹음의 서로 다른 발화가 Golden과 Replay에 동시에 들어가지 않으며, Audio가 없는 라벨은 학습·평가 기준 세트에 포함되지 않습니다.

| 세트 | 원본 녹음 수 | 발화 수 | 용도 | GCS 매니페스트 |
| --- | ---: | ---: | --- | --- |
| Golden v3 | 15 | 300 | 기존 성능 회귀 방지용 평가 전용 | `flywheel/stt/eval/old-golden-v3.jsonl` |
| Replay v3 | 60 | 3,000 | 증분 학습 시 과거 데이터 재노출 | `flywheel/stt/replay-v3.jsonl` |

두 매니페스트는 create-only로 작성되어 덮어쓰지 않습니다. 원본 Audio와 Label JSON도 수정하지 않습니다. v1·v2 매니페스트는 초기 실험 산출물로 보존하지만 구성 파일에서 참조하지 않으며, Audio 존재 검증을 거친 v3만 운영 기준으로 사용합니다.

Golden은 매 Flywheel 실행마다 운영·후보 모델을 모두 평가하므로 300발화로 제한합니다. Replay는 매 학습에 신규 Train 수만큼만 결정론적으로 추출하므로, 3,000발화를 저장해도 소규모 증분 학습의 비용은 증가하지 않고 과거 데이터의 다양성만 커집니다.

## 스냅샷과 중복 실행 방지

현재 테스트 임계값은 승인·미학습 데이터 **10개**입니다. `snapshot`은 다음을 수행합니다.

1. `status=approved` 또는 `review_status=approved`이고 `training_status.stt.promoted=false`인 API 라벨을 읽습니다.
2. 데이터 구성과 GCS 객체 generation을 해시해 `snapshot_id`를 만듭니다.
3. `flywheel/stt/snapshots/{snapshot_id}/manifest.jsonl` 및 `runs/{snapshot_id}/run.json`을 create-only로 저장합니다.
4. 동일 스냅샷이 이미 있으면 새 학습을 시작하지 않습니다.

현재 생성된 첫 테스트 스냅샷은 `9f93283c3c966eaf2d99`이며, 승인·미학습 발화 12개를 포함합니다. 스냅샷 생성 자체는 학습·배포를 수행하지 않습니다.

## 학습 및 평가 흐름

```text
승인 신규 데이터 → 불변 스냅샷 → 신규 Train / 신규 Holdout 분리
                                      ↓
현재 운영 LoRA + Replay → 후보 LoRA 학습
                                      ↓
운영 LoRA와 후보 LoRA를 Golden WER / 신규 Holdout WER로 비교
                                      ↓
통과 시 백업 → 운영 경로 교체 → source JSON promoted=true
```

- 학습은 기존 `train.py` 방식처럼 발화 구간의 음성을 로드하고 Whisper feature를 배치에서 생성합니다.
- LoRA만 증분 업데이트하며 기본 설정은 batch 4, gradient accumulation 4, learning rate `1e-4`, 2 epoch입니다.
- 신규 데이터는 결정론적으로 80% Train, 20% Holdout으로 나뉩니다.
- Replay 비율은 신규 Train 수와 동일한 수까지 사용합니다. 따라서 아주 작은 테스트 실행에서도 전체 Replay 1,000개를 매번 모두 학습하지 않습니다.

## 승격 규칙과 롤백

후보 가중치는 다음 두 조건을 모두 만족해야 승격됩니다.

| 지표 | 조건 | 목적 |
| --- | --- | --- |
| Golden WER | 기존 운영 모델보다 악화가 0.5%p 이하 | 기존 입력 성능 저하 방지 |
| 신규 Holdout WER | 기존 운영 모델보다 최소 1.0%p 개선 | 새 데이터 학습의 실효성 확인 |

통과하면 다음 순서로 처리합니다.

1. 현재 `whisper-jeju-lora-final/` 전체를 `whisper-model-weights/backups/stt-{snapshot_id}/`에 먼저 백업합니다.
2. 후보 LoRA를 `whisper-model-weights/whisper-jeju-lora-final/`로 복사합니다.
3. `flywheel/stt/releases/current.json`에 후보·백업 경로와 평가 보고서를 기록합니다.
4. 해당 스냅샷의 source JSON에만 `promoted=true`와 학습 시각을 기록합니다.

평가 실패 시 운영 가중치와 source JSON은 변경되지 않습니다. 롤백이 필요하면 release record의 `backup_adapter_uri`를 운영 경로로 복사하면 됩니다.

## 실행 명령

```bash
# 기존 API 라벨에 training_status 추가
python -m flywheel.cli --config configs/stt.yaml migrate-status --dry-run
python -m flywheel.cli --config configs/stt.yaml migrate-status

# Golden/Replay v3 생성 (기본: 300 / 3,000 발화, 실제 Audio 존재 검증)
python -m flywheel.cli --config configs/stt.yaml bootstrap-baseline

# 승인·미학습 데이터가 10개 이상이면 스냅샷 생성
python -m flywheel.cli --config configs/stt.yaml snapshot

# 특정 스냅샷의 Cloud Run 학습은 GitHub Actions 또는 Cloud Run Job으로 실행
python -m flywheel.pipeline 9f93283c3c966eaf2d99 --config configs/stt.yaml --promote
```

## Cloud Run 및 GitHub Actions

- `deploy-stt-flywheel.yml`: Docker 이미지를 Artifact Registry에 올리고 GPU(L4) Cloud Run Job을 배포합니다.
- `snapshot-stt-flywheel.yml`: 매일 실행하거나 수동 실행해 스냅샷을 만들고, 신규 스냅샷일 때만 학습 Job을 실행합니다.
- `run-stt-flywheel.yml`: 특정 snapshot ID를 지정해 수동 재실행합니다.

GitHub Actions에는 Workload Identity Federation을 사용합니다. 필요한 repository secrets는 다음과 같습니다.

- `GCP_PROJECT_ID`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_FLYWHEEL_SERVICE_ACCOUNT` (GitHub Actions 배포·실행 계정)
- `GCP_STT_RUNTIME_SERVICE_ACCOUNT` (Cloud Run Job 런타임 계정)

런타임 계정에는 데이터 버킷 및 LoRA 경로의 읽기·쓰기 권한이 필요합니다. GitHub Actions 계정에는 Artifact Registry, Cloud Run Job 배포·실행 권한이 필요합니다.
