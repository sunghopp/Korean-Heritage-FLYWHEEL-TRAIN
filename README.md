# Korean Heritage Data Flywheel

![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python&logoColor=white) ![Google Cloud](https://img.shields.io/badge/Google%20Cloud-Run%20Jobs%20%7C%20GCS-4285F4?logo=googlecloud&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-2088ff?logo=githubactions&logoColor=white)

> 승인된 제주어 데이터로 STT LoRA와 번역 모델을 증분 학습하고, 평가를 통과한 후보만 승격하는 파이프라인입니다.

## 프로젝트 목적

API가 수집한 음성·라벨과 사람이 승인한 번역 쌍을 GCS snapshot으로 고정해 학습합니다. 기존 기준에서 퇴행하거나 새 데이터에서 충분히 개선되지 않으면 운영 모델을 바꾸지 않습니다.

## 핵심 기능

- STT/Gemini snapshot을 별도로 만들고 GCS generation 기반 ID와 lock으로 중복 실행을 줄입니다.
- Whisper Small LoRA 후보를 Golden·Holdout WER로 평가합니다.
- 사람이 승인한 제주어→표준어 데이터로 기존 tuned Gemini를 이어 학습하고 Golden·Validation BLEU로 평가합니다.
- STT 오디오와 Gemini 번역 replay를 학습에 다시 노출합니다.
- 통과한 STT adapter는 기존 prefix 교체 전 백업하고, Gemini release는 GCS에 기록합니다.
- GitHub Actions 수동 workflow로 snapshot, Cloud Run Job, tuning, 배포를 진행합니다.

## 아키텍처

~~~mermaid
flowchart TD
  API[API 음성·라벨] --> GCS[GCS 데이터·검수]
  HUMAN[사람 검수] --> GCS
  GCS --> SNAP[Actions snapshot]
  SNAP -->|min_samples| MAN[불변 manifest·run]
  MAN --> STT[STT Cloud Run Job]
  STT --> WER[Golden·Holdout WER]
  WER -->|통과| ADAPTER[adapter backup·운영 prefix 교체]
  MAN --> GEM[Gemini Cloud Run Job]
  GEM --> BLEU[Golden·Validation BLEU]
  BLEU -->|통과| REL[release·current 기록]
  REL --> ACT[Actions가 API endpoint 갱신]
  ADAPTER --> API
~~~

설정은 configs/stt.yaml과 configs/gemini.yaml이며 데이터·상태·산출물은 GCS에 저장합니다. Firestore는 사용하지 않습니다.

## 기술 스택

Python 3.11, PyTorch, Transformers, PEFT, Whisper, jiwer, Google Gen AI SDK, Agent Platform tuning API, sacreBLEU, Google Cloud Storage, Docker, Cloud Run Jobs, GitHub Actions.

## 설치 및 실행

Python 3.11과 Google Cloud ADC가 필요합니다.

~~~bash
python -m pip install -r requirements-control.txt
python -m pip install -r requirements-gemini.txt
python -m flywheel.cli --config configs/stt.yaml snapshot
python -m flywheel.gemini_cli --config configs/gemini.yaml snapshot
~~~

STT job 이미지 빌드·실행:

~~~bash
docker build -t jeju-stt-flywheel .
docker run --rm jeju-stt-flywheel <snapshot_id> --config configs/stt.yaml --promote
~~~

Gemini 이미지:

~~~bash
docker build -f cloudrun/Dockerfile.gemini -t jeju-gemini-flywheel .
~~~

운영 배포와 실행은 GitHub Actions workflow_dispatch에서 합니다. Gemini는 submit, wait, finalize로 재개할 수 있습니다.

## 환경 변수

프로세스 환경 변수 대신 YAML 설정을 사용합니다.

| 파일 | 설정 |
|---|---|
| configs/stt.yaml | GCS 경로, Whisper 모델, 분할, WER gate |
| configs/gemini.yaml | GCS 경로, 초기 endpoint/model, validation, BLEU gate |

Actions secrets: GCP_PROJECT_ID, GCP_WORKLOAD_IDENTITY_PROVIDER, GCP_FLYWHEEL_SERVICE_ACCOUNT, GCP_STT_RUNTIME_SERVICE_ACCOUNT. 서비스 계정 키 JSON은 사용하지 않습니다. 각 계정에 GCS·Artifact Registry·Cloud Run·Agent Platform 권한과 로컬 ADC가 필요합니다.

## API 및 데이터 흐름

STT는 dataset/extracted/Text의 승인된 미학습 라벨과 오디오를 사용합니다. Gemini는 status=approved, reviewed_by=human, training_status.gemini.promoted=false인 번역 쌍만 사용합니다. 현재 두 config의 min_samples는 10입니다.

~~~text
입력 → snapshot → train/holdout 또는 validation 분리 + replay
     → Cloud Run 학습·튜닝 → Golden 및 새 데이터 평가 → 승인 후보 release
~~~

| 산출물 | STT | Gemini |
|---|---|---|
| Snapshot | flywheel/stt/snapshots/{id}/manifest.jsonl | flywheel/gemini/snapshots/{id}/manifest.jsonl |
| Run | flywheel/stt/runs/{id}/run.json | flywheel/gemini/runs/{id}/run.json |
| Evaluation | flywheel/stt/evaluations/{id}.json | flywheel/gemini/evaluations/{id}.json |
| Current release | flywheel/stt/releases/current.json | flywheel/gemini/releases/current.json |

Gemini API endpoint 반영은 Run Gemini Flywheel workflow가 평가 결과를 확인한 뒤 처리합니다.

## 디렉터리 구조

~~~text
.
├── .github/workflows/
├── cloudrun/
├── configs/stt.yaml
├── configs/gemini.yaml
├── flywheel/cli.py
├── flywheel/gemini_cli.py
├── flywheel/pipeline.py
├── flywheel/gemini_pipeline.py
├── flywheel/train_stt.py
├── flywheel/evaluate_stt.py
├── flywheel/gemini_flywheel.py
├── Dockerfile
├── cloudrun/Dockerfile.gemini
└── requirements*.txt
~~~

## 학습 및 평가 지표

| Pipeline | 기준 규모·분할 | 승격 조건 |
|---|---|---|
| STT | Golden 300 발화, Replay 3,000 발화, 신규 80% train / 20% holdout | Golden WER 퇴행 ≤ 0.5%p, Holdout WER 개선 ≥ 1.0%p |
| Gemini | Golden 300쌍, Replay 3,000쌍, 최소 10개 시 validation 2개 | Golden BLEU 하락 ≤ 1.0점, Validation BLEU 개선 ≥ 1.0점 |

현재 config 임계값은 시험용입니다. Gemini는 Exact Match Rate도 기록하나 승격은 0–100 corpus BLEU로 판단합니다. 발표 자료의 카카오브레인 데이터 200개 샘플 평가는 별도 기준이며 수치 점수는 없습니다. 실제 회차 결과는 GCS evaluation JSON에 기록됩니다.

## 주의사항

- min_samples 10은 시험용 설정입니다. 운영 전에 두 config를 함께 조정하세요.
- STT 승격은 운영 adapter GCS prefix를 교체하고 API 재시작이 필요합니다.
- STT workflow가 설정하는 STT_RELEASE_MANIFEST_PATH는 현재 API가 읽지 않습니다. API의 실제 STT 경로는 LORA_MODEL_PATH입니다.
- Gemini job은 최대 4시간 기다리며, 미완료 시 finalize로 재개합니다.
- Golden/Replay manifest는 덮어쓰지 않게 생성됩니다. Actions는 GCP 리소스를 실행하므로 필요한 권한을 구성하세요.
