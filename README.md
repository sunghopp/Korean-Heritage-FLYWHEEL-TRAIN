# Korean Heritage Data Flywheel

제주어 서비스의 **STT(음성→텍스트)** 와 **Gemini 번역(제주어→표준어)** 을 안전하게 증분 학습하는 GCS 전용 파이프라인입니다. API가 수집·승인한 데이터를 재사용하되, 기존 사용자 입력 성능이 나빠지면 새 모델을 운영에 반영하지 않습니다.

> **현재 테스트 설정** — 두 파이프라인 모두 승인·미학습 데이터가 `min_samples: 10`개 이상일 때 시작합니다. 운영 전에는 `configs/stt.yaml`, `configs/gemini.yaml`의 같은 파라미터를 함께 상향하세요.

## 한눈에 보기

```text
Web / API
  └─ GCS Audio + 기존 형식 JSON 저장 · 검수/승인
       └─ Start Data Flywheel Training (GitHub Actions)
            ├─ STT: snapshot → LoRA 학습 → WER 평가 → 승인 시 가중치 교체
            └─ Gemini: snapshot → 기존 조정 모델 기반 SFT → 완료 대기
                       → BLEU 평가 → 승인 시 번역 endpoint 교체
```

| 구분 | STT | Gemini 번역 |
| --- | --- | --- |
| 학습 대상 | 음성·전사 라벨 | `dialect_form` → `standard_form` |
| 실행 환경 | GPU(L4) Cloud Run Job | CPU Cloud Run Job + Agent Platform tuning |
| 기존 모델 보호 | Golden/New Holdout WER | Golden/New Validation BLEU |
| 승인 시 반영 | 운영 LoRA manifest 갱신 | `GEMINI_TUNED_ENDPOINT` 갱신 |
| 망각 방지 | 기존 Replay 음성 재학습 | 기존 Replay 번역 쌍 재노출 |

## 범위와 책임

| 구성 요소 | 담당 책임 | 저장소·경로 |
| --- | --- | --- |
| API 서버 | 음성·JSON 생성, 기존 Confidence 승인 로직 | `Korean-Heritage-API-SVR` |
| 이 저장소 | 스냅샷, 학습, 평가, 승격, release 기록 | `Korean-Heritage-FLYWHEEL-TRAIN` |
| 신규 데이터 | API가 만드는 Audio·Text JSON | `dataset/extracted/Audio`, `dataset/extracted/Text` |
| STT 기준 데이터 | 기존 고정 Audio·Label | `extracted/extracted/Audio/Audio`, `extracted/extracted/Text/Text/Label` |
| 운영 STT LoRA | 현재 API가 참조하는 어댑터 | `whisper-model-weights/whisper-jeju-lora-final/` |

Firestore는 사용하지 않습니다. 상태, 스냅샷, 평가 결과, 릴리스 기록은 GCS에만 저장합니다.

## JSON 호환성과 학습 상태

기존 학습용 JSON의 음성·전사·번역·Confidence 필드는 변경하지 않습니다. Flywheel은 아래 `training_status`만 추가해 같은 JSON을 STT와 Gemini가 독립적으로 한 번씩 학습할 수 있게 합니다.

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

`promoted: false`인 항목만 새 snapshot 후보입니다. 학습이나 평가가 실패하면 `false`를 유지하므로 사람이 보정한 데이터는 다음 실행에서 다시 사용합니다. 입력 JSON URI와 GCS generation을 해시한 `snapshot_id`, create-only 객체, 활성 lock으로 동일 snapshot의 중복 실행도 막습니다.

## 공통 시작 임계값

두 Flywheel은 각 설정 파일의 동일한 `min_samples` 파라미터를 사용합니다.

| 설정 파일 | 현재 값 | 의미 |
| --- | ---: | --- |
| `configs/stt.yaml` | `min_samples: 10` | 승인·미학습 STT 라벨 최소 개수 |
| `configs/gemini.yaml` | `min_samples: 10` | 사람 승인·미학습 번역 쌍 최소 개수 |

Gemini 테스트에서는 10개 중 2개를 Validation, 8개를 신규 Train으로 분리하고 Replay를 최대 8개 추가합니다. 운영에서 100개로 올릴 경우 Gemini의 `min_validation_samples`도 20으로 함께 조정합니다.

## STT Flywheel

### 데이터·기준 세트

STT snapshot은 `status=approved` 또는 `review_status=approved`이면서 `training_status.stt.promoted=false`인 JSON을 사용합니다. 기존 기준 데이터에서 실제 Audio가 존재하는 항목만 아래 불변 매니페스트에 담습니다.

| 세트 | 원본 녹음 수 | 발화 수 | 용도 | GCS 매니페스트 |
| --- | ---: | ---: | --- | --- |
| Golden v3 | 15 | 300 | 기존 성능 회귀 감시 전용 | `flywheel/stt/eval/old-golden-v3.jsonl` |
| Replay v3 | 60 | 3,000 | 과거 음성 재노출용 풀 | `flywheel/stt/replay-v3.jsonl` |

Golden과 Replay는 create-only로 생성되고 서로 같은 녹음이 섞이지 않습니다. 매 회차에는 전체 Replay를 학습하지 않고 신규 Train 수와 같은 수까지만 결정론적으로 뽑아 비용을 제한합니다.

### 학습·승격 흐름

```text
승인 신규 STT 데이터
  → 불변 snapshot
  → 신규 Train 80% / Holdout 20% 분리
  → 현재 운영 LoRA + Replay로 증분 LoRA 학습
  → 운영·후보의 Golden WER와 신규 Holdout WER 비교
  → 통과: 기존 LoRA 백업 → 운영 LoRA 교체 → JSON stt.promoted=true
```

Whisper 학습은 기존 `train.py`처럼 발화 구간 Audio에서 feature를 만들고 LoRA만 업데이트합니다. 기본값은 batch 4, gradient accumulation 4, learning rate `1e-4`, 2 epoch입니다.

| 평가 지표 | 승인 조건 | 보호 대상 |
| --- | --- | --- |
| Golden WER | 운영 모델보다 악화가 최대 0.5%p | 기존 입력 성능 |
| 신규 Holdout WER | 운영 모델보다 최소 1.0%p 개선 | 새 입력 학습 효과 |

통과 시 기존 운영 경로를 `whisper-model-weights/backups/stt-{snapshot_id}/`에 먼저 백업한 후 후보를 `whisper-jeju-lora-final/`에 반영합니다. release record의 `backup_adapter_uri`로 이전 가중치를 복원할 수 있습니다.

## Gemini 제주어 번역 Flywheel

### 학습 범위와 JSONL 형식

Gemini는 **제주어 문장(`dialect_form`)을 표준어(`standard_form`)로 번역**하는 작업만 조정합니다. `ars_reply_jeju`, Scenario, 음성, Confidence는 Gemini 학습 입력에 넣지 않습니다. Scenario는 추후 RAG가 담당합니다.

학습 JSONL은 기존 조정 데이터와 동일한 구조입니다.

```json
{"systemInstruction":{"role":"system","parts":[{"text":"당신은 제주 방언을 표준어로 정확하게 번역하는 전문 번역가입니다."}]},"contents":[{"role":"user","parts":[{"text":"죽어도 안 된댄."}]},{"role":"model","parts":[{"text":"죽어도 안 된대."}]}]}
```

기본적으로 `status=approved`, `reviewed_by=human`, `training_status.gemini.promoted=false`인 번역 쌍만 사용합니다. 자동 승인 결과를 모델의 정답으로 되먹이면 자기 검증이 되기 때문에 기본 BLEU gate에서는 제외합니다.

### 기존 조정 모델을 보존하는 연속 학습

첫 튜닝은 Agent Platform의 기존 모델 version `880260762860257280@1`을 `preTunedModel`로 지정해 시작합니다. 새 base model에서 처음부터 학습하는 것이 아닙니다. 승인 후보의 `candidate_model`은 `flywheel/gemini/releases/current.json`에 기록되고, 다음 회차는 그 후보를 다시 `preTunedModel`로 사용합니다.

```text
기존 번역 모델 v1
  → 후보 v2 평가·승인
  → v2를 다음 회차의 preTunedModel로 사용
  → 후보 v3 평가·승인 …
```

각 회차는 새 model version/endpoint를 만들므로 이전 endpoint는 유지됩니다. 평가 실패 시 API는 기존 endpoint를 계속 사용합니다.

### Golden·Replay·BLEU 평가

`valid_gemini_jeju.jsonl`에서 최초 한 번 생성하는 기준 세트입니다.

| 세트 | 개수 | 역할 | 경로 |
| --- | ---: | --- | --- |
| Golden v1 | 300 | 기존 번역 성능 회귀 감시 | `flywheel/gemini/eval/old-golden-v1.jsonl` |
| Replay v1 | 3,000 | 기존 번역 능력 망각 완화 | `flywheel/gemini/replay-v1.jsonl` |

번역 생성은 고정 system instruction과 `temperature=0`으로 수행합니다. Exact Match Rate도 기록하지만, 승인 판단은 0~100 scale의 corpus BLEU만 사용합니다.

| 평가 세트 | 기록 지표 | 승인 조건 |
| --- | --- | --- |
| 기존 Golden 300개 | BLEU, Exact Match Rate | BLEU 하락이 최대 1.0점 |
| 신규 Validation | BLEU, Exact Match Rate | BLEU가 최소 1.0점 상승 |

둘 다 통과하면 `GEMINI_TUNED_ENDPOINT`를 후보 endpoint로 바꾸고, 실패하면 기존 endpoint를 유지합니다. 이전 endpoint와 후보 endpoint, source/candidate model, BLEU 결과는 release record에 저장됩니다.

## GCS 산출물과 확인 위치

| 종류 | STT 경로 | Gemini 경로 |
| --- | --- | --- |
| 입력 snapshot | `flywheel/stt/snapshots/{id}/manifest.jsonl` | `flywheel/gemini/snapshots/{id}/manifest.jsonl` |
| 실행 상태 | `flywheel/stt/runs/{id}/run.json` | `flywheel/gemini/runs/{id}/run.json` |
| 평가 보고서 | `flywheel/stt/evaluations/{id}.json` | `flywheel/gemini/evaluations/{id}.json` |
| 현재 릴리스 | `flywheel/stt/releases/current.json` | `flywheel/gemini/releases/current.json` |

`run.json`에는 snapshot ID, 입력 개수, 학습 상태, 튜닝 Job ID와 대기 상태가 남고, `evaluations/*.json`에는 운영/후보 모델의 지표와 `approved` 결과가 남습니다.

## GitHub Actions 사용법

모든 Action은 Workload Identity Federation으로 GCP에 인증합니다. 서비스 계정 키 파일은 GitHub에 저장하지 않습니다.

필요한 repository secrets:

- `GCP_PROJECT_ID`
- `GCP_WORKLOAD_IDENTITY_PROVIDER`
- `GCP_FLYWHEEL_SERVICE_ACCOUNT` — GitHub Actions의 배포·실행 계정
- `GCP_STT_RUNTIME_SERVICE_ACCOUNT` — Cloud Run Job 런타임 계정

### Action별 역할

| Action | 트리거 | 하는 일 |
| --- | --- | --- |
| `Deploy STT Flywheel Job` | `main` push 또는 수동 | STT GPU Docker image를 빌드하고 `jeju-stt-flywheel` Job 정의 갱신 |
| `Deploy Gemini Flywheel Job` | `main` push 또는 수동 | Gemini CPU Docker image를 빌드하고 `jeju-gemini-flywheel` Job 정의 갱신 |
| `Start Data Flywheel Training` | 수동 | `both`/`stt`/`gemini`를 선택해 snapshot 생성부터 학습·평가까지 시작 |
| `Run STT Flywheel` | 수동 | 특정 STT snapshot을 재실행 |
| `Run Gemini Flywheel` | 수동 | 특정 Gemini snapshot을 `submit`, `wait`, `finalize` 모드로 재개 |
| `Bootstrap Gemini Golden and Replay` | 수동·최초 1회 | 기존 `valid_gemini_jeju.jsonl`에서 Golden/Replay 생성 |

정기 schedule은 비용 통제를 위해 주석 처리되어 있습니다. 현재는 **Actions → Run workflow**로만 시작합니다.

### 일반 실행 순서

1. PR merge 후 `Deploy STT Flywheel Job`, `Deploy Gemini Flywheel Job` 성공을 확인합니다. `main` merge는 두 배포 Action을 자동 실행합니다.
2. Gemini Golden/Replay가 없다면 `Bootstrap Gemini Golden and Replay`를 한 번 실행하고, GCS에 업로드한 `valid_gemini_jeju.jsonl` URI를 입력합니다.
3. **Start Data Flywheel Training**에서 `both` 또는 필요한 pipeline을 선택합니다.
4. STT는 GPU Job에서 학습·WER 평가·승격까지 끝냅니다.
5. Gemini는 CPU Job이 tuning 제출 뒤 5분 간격으로 Agent Platform 상태를 확인하며 최대 4시간 기다립니다.
6. Gemini tuning이 완료되면 같은 실행에서 BLEU 평가와 endpoint 반영까지 진행됩니다.

Gemini tuning이 4시간을 넘기면 Cloud Run Job은 정상 종료하고 운영 endpoint를 유지합니다. 완료 뒤 **Run Gemini Flywheel**의 `finalize`를 실행하면 즉시 상태 확인·평가를 재개할 수 있고, `wait`는 다시 최대 4시간 대기합니다.

## 로컬 점검 명령

```bash
# 기존 API JSON에 training_status 추가: 먼저 dry-run 권장
python -m flywheel.cli --config configs/stt.yaml migrate-status --dry-run
python -m flywheel.cli --config configs/stt.yaml migrate-status

# STT Golden/Replay v3 생성
python -m flywheel.cli --config configs/stt.yaml bootstrap-baseline

# Gemini Golden 300 / Replay 3,000 생성
python -m flywheel.gemini_cli --config configs/gemini.yaml bootstrap-baseline \
  --source-uri="gs://<bucket>/<valid_gemini_jeju.jsonl>" --golden-count=300 --replay-count=3000

# snapshot만 로컬에서 확인
python -m flywheel.cli --config configs/stt.yaml snapshot
python -m flywheel.gemini_cli --config configs/gemini.yaml snapshot
```

## 권한 요약

Cloud Run 런타임 계정에는 GCS 객체 읽기·쓰기 권한과 Agent Platform tuning job 생성·조회 권한(일반적으로 `roles/aiplatform.user`)이 필요합니다. GitHub Actions 계정에는 Artifact Registry, Cloud Run Job 배포·실행, API 환경변수 갱신 권한이 필요합니다.
