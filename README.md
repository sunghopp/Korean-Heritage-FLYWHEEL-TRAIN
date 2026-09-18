# Korean Heritage Data Flywheel

제주어 Whisper LoRA의 증분 학습을 위한 GCS 전용 Data Flywheel 프로젝트입니다. API 서버와 학습 Pipeline의 책임을 분리하며, Firestore는 사용하지 않습니다.

## 목적과 범위

- Web/API에서 수집·승인된 음성 및 JSON을 누적하고, 일정 수량이 되면 Cloud Run Job에서 증분 LoRA 학습을 수행합니다.
- 기존 사용자 입력 성능 저하를 막기 위해 기존 Golden Set과 신규 Holdout Set의 WER을 모두 비교한 뒤에만 가중치를 교체합니다.
- 전체 재학습 대신 현재 운영 LoRA에서 시작하고, 과거 Replay 데이터를 섞어 학습 비용과 망각을 줄입니다.
- STT LoRA와 Gemini 제주어→표준어 번역 조정을 각각 독립된 Cloud Run Job으로 운영합니다. Scenario/ARS 답변은 조정 데이터에 넣지 않으며, 추후 RAG가 담당합니다.

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
- `snapshot-stt-flywheel.yml`: STT와 Gemini가 같은 수동 Trigger에서 스냅샷을 만들고, 신규 스냅샷일 때만 각 학습 Job을 시작합니다.
- `run-stt-flywheel.yml`: 특정 snapshot ID를 지정해 수동 재실행합니다.

GitHub Actions에는 Workload Identity Federation을 사용합니다. 필요한 repository secrets는 다음과 같습니다.

- `GCP_PROJECT_ID`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_FLYWHEEL_SERVICE_ACCOUNT` (GitHub Actions 배포·실행 계정)
- `GCP_STT_RUNTIME_SERVICE_ACCOUNT` (Cloud Run Job 런타임 계정)

런타임 계정에는 데이터 버킷 및 LoRA 경로의 읽기·쓰기 권한이 필요합니다. GitHub Actions 계정에는 Artifact Registry, Cloud Run Job 배포·실행 권한이 필요합니다.

## Gemini 제주어 번역 Flywheel

### 범위와 안전한 릴리스 원칙

Gemini 파이프라인은 **제주어 입력(`dialect_form`) → 표준어 번역(`standard_form`)** 한 가지 작업만 학습합니다. 기존 JSON 경로와 형식은 그대로 사용하며, `training_status.gemini`만 상태 기록에 사용합니다. `ars_reply_jeju`, Scenario, 음성 파일, Confidence 산출값은 Gemini 조정 학습 데이터로 저장하거나 사용하지 않습니다.

기존 Gemini 조정 모델도 번역 쌍만 학습했다는 전제에서, 후보가 BLEU gate를 통과하면 `Run Gemini Flywheel`의 `finalize` 단계가 API의 `GEMINI_TUNED_ENDPOINT`를 후보 endpoint로 즉시 교체합니다. 이전 endpoint와 후보 endpoint는 release record에 함께 남으므로, 이전 endpoint로 API 환경변수를 되돌려 롤백할 수 있습니다.

사용자가 지정한 Agent Platform 모델 version `880260762860257280@1`은 첫 **연속 조정 시작점**입니다. 이후에는 직전 승인 모델 version을 `preTunedModel`로 사용해 기존 LoRA/조정 가중치 위에 추가 SFT를 수행합니다. 각 조정은 새 version/endpoint를 생성하며, 기존 endpoint와 후보 endpoint를 같은 데이터로 비교한 뒤 후보를 릴리스합니다.

### GCS 상태와 불변 산출물

Firestore는 사용하지 않습니다. 스냅샷의 입력 JSON URI와 GCS generation을 해시한 `snapshot_id`와 create-only GCS 객체가 중복 실행을 막습니다.

| 산출물 | 경로 | 역할 |
| --- | --- | --- |
| 신규 스냅샷 | `flywheel/gemini/snapshots/{snapshot_id}/manifest.jsonl` | 승인된 신규 번역 쌍의 불변 입력 목록 |
| 튜닝 Train/Validation | `.../train.jsonl`, `.../validation.jsonl` | Gemini tuning API가 읽는 JSONL |
| 실행 상태 | `flywheel/gemini/runs/{snapshot_id}/run.json` | created/prepared/submitted/approved/rejected 및 tuning job ID |
| 기존 Golden | `flywheel/gemini/eval/old-golden-v1.jsonl` | 기존 번역 성능 회귀 평가 |
| Replay | `flywheel/gemini/replay-v1.jsonl` | 증분 튜닝 시 과거 번역 재노출 |
| 평가/릴리스 | `flywheel/gemini/evaluations/{snapshot_id}.json`, `releases/*.json` | BLEU 추이, 후보 endpoint, 승인 결과 |

같은 snapshot ID의 manifest/run이 이미 존재하면 새 튜닝 Job을 만들지 않습니다. 후보가 거절되면 운영 엔드포인트와 source JSON은 바뀌지 않으므로, 사람이 보정한 데이터는 다음 회차에 다시 사용할 수 있습니다.

### 데이터 선정과 분할

기본값은 사람이 검수한 `status=approved` 및 `reviewed_by=human` 데이터 중 `training_status.gemini.promoted=false`인 항목만 사용합니다. Confidence 자동 승인 결과를 다시 그 모델의 정답처럼 평가하면 자기 검증이 되므로, 자동 승인 데이터는 운영 저장에는 남아도 기본 학습·BLEU gate에는 포함하지 않습니다.

현재 임계값은 신규 **100개**입니다. 100개 도달 시 해시 기반으로 20개를 신규 Validation, 80개를 신규 Train으로 고정 분리합니다. Train에는 Replay를 최대 80개(신규 Train과 1:1)만 추가합니다. 따라서 전체 3,000개 Replay를 매번 튜닝하지 않아 비용을 제한하면서도 과거 번역을 잊는 위험을 낮춥니다.

`valid_gemini_jeju.jsonl`은 먼저 GCS에 업로드한 뒤 Golden 300개와 Replay 3,000개를 결정론적으로, 서로 겹치지 않게 생성합니다. 두 세트는 create-only이므로 버전별 BLEU가 같은 기준을 유지합니다.

### 평가와 승격 기준

모든 번역 결과는 고정된 번역 system instruction과 `temperature=0`으로 생성합니다. 다음 지표를 매 회차 GCS 평가 JSON에 기록합니다.

| 평가 세트 | 기록 지표 | 승인 조건 |
| --- | --- | --- |
| 기존 Golden 300개 | corpus BLEU, Exact Match Rate | 기존 기준선 대비 BLEU 하락이 1.0점 이하여야 함 |
| 신규 Validation 최소 20개 | corpus BLEU, Exact Match Rate | 기존 기준선 대비 BLEU가 최소 1.0점 향상되어야 함 |

BLEU는 번역 전체의 n-gram 유사도를 일관된 0~100 점수로 비교하고, Exact Match Rate는 사람이 읽을 때의 보조 해석 지표입니다. 승인 판정은 BLEU만 사용합니다. 둘 중 하나라도 실패하면 `approved=false`이고 기존 모델은 유지됩니다.

### 실행 순서

자동 스케줄은 STT와 Gemini 모두 주석 처리되어 있습니다. GitHub Actions의 **Run workflow**에서만 실행됩니다.

1. `Bootstrap Gemini Golden and Replay`를 한 번 실행합니다. `valid_gemini_jeju.jsonl`을 올린 `gs://...` URI를 입력합니다.
2. `Deploy Gemini Flywheel Job`으로 CPU Cloud Run Job `jeju-gemini-flywheel`을 배포합니다.
3. `Start Data Flywheel Training`을 실행합니다. `both`는 STT·Gemini를 함께, `stt`/`gemini`는 해당 파이프라인만 시작합니다. 두 파이프라인은 같은 수동 Trigger와 비활성화된 동일 스케줄 정책을 사용합니다.
4. Gemini 신규 데이터가 100개 이상이면 이 Trigger가 스냅샷을 만들고 tuning job을 제출합니다. 제출 Job은 ID만 GCS에 기록하므로 완료 대기 비용이 없습니다.
5. Agent Platform에서 tuning이 완료된 뒤 `Run Gemini Flywheel`을 `mode=finalize`로 실행합니다. Golden/New BLEU를 비교해 통과하면 release record를 생성하고 API의 `GEMINI_TUNED_ENDPOINT`를 즉시 후보 endpoint로 교체합니다. 거절 시 API endpoint는 변경하지 않습니다.

### 권한과 구성

Gemini Job은 기존 `GCP_STT_RUNTIME_SERVICE_ACCOUNT`를 런타임 계정으로 재사용합니다. 이 계정에는 GCS 객체 읽기/쓰기 외에 Agent Platform tuning job 생성·조회 권한(통상 `roles/aiplatform.user`)이 필요합니다. GitHub Actions 계정은 기존 Workload Identity Federation의 Artifact Registry·Cloud Run 배포/실행 권한을 계속 사용합니다.

연속 조정은 Agent Platform REST API의 `preTunedModel`을 사용합니다. 첫 실행은 `configs/gemini.yaml`의 지정 모델 version에서 시작하며, 이후 실행은 `releases/current.json`의 승인된 `candidate_model`을 자동으로 다음 시작점으로 사용합니다.
