from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Literal

import librosa
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from google import genai
from google.api_core.exceptions import NotFound
from google.genai import types
import agentplatform
from agentplatform import rag
from peft import PeftConfig, PeftModel
from pydantic import BaseModel, Field
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from ars_prompt import SYSTEM_INSTRUCTION, build_few_shot_contents
from dataset_dashboard import (
    get_audio_bytes,
    get_stats,
    list_samples,
    update_sample_label,
)
from dataset_logger import save_training_sample
from gcs_model_loader import load_lora_model_path
from tts_engine import JejuVITSEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ==========================================
# 1. FastAPI / environment
# ==========================================
app = FastAPI(title="Jeju AI ARS API", version="2.0.0")

cors_origins = [x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

LORA_MODEL_PATH = load_lora_model_path()
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "385248657749")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_TUNED_ENDPOINT = os.getenv(
    "GEMINI_TUNED_ENDPOINT",
    "projects/385248657749/locations/us-central1/endpoints/7571681821318971392",
)

TTS_CONFIG_PATH = os.getenv("TTS_CONFIG_PATH", "./tts_config/jeju_vits.json")
TTS_CHECKPOINT_PATH = os.getenv(
    "TTS_CHECKPOINT_PATH",
    "gs://malmoi-jeju-dataset-2026/tts/jeju_vits.pth",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFERENCE_LOCK = asyncio.Lock()


class GeminiARSResult(BaseModel):
    standard_text: str = Field(description="입력 제주어를 자연스러운 표준어로 번역한 결과")
    ars_reply_jeju: str = Field(description="Demo 시나리오를 참고해 생성한 제주어 AI ARS 답변")


class ConversationTurn(BaseModel):
    """One completed user/assistant turn supplied by the demo browser."""

    jeju_text: str = Field(min_length=1, max_length=500)
    standard_text: str = Field(min_length=1, max_length=500)
    ars_reply_jeju: str = Field(min_length=1, max_length=1_000)


class TTSRequest(BaseModel):
    text: str


# ==========================================
# 2. Models: load once at process startup
# ==========================================
print(f"서버 구동 준비: STT 모델 적재 중... device={DEVICE}")
print(f"STT 모델 경로: {LORA_MODEL_PATH}")
stt_config = PeftConfig.from_pretrained(LORA_MODEL_PATH)
base_model_name = stt_config.base_model_name_or_path
processor = WhisperProcessor.from_pretrained(base_model_name)
base_model = WhisperForConditionalGeneration.from_pretrained(base_model_name)
stt_model = PeftModel.from_pretrained(base_model, LORA_MODEL_PATH).to(DEVICE).eval()

print("Gemini client 준비 중...")
gemini_client = genai.Client(
    vertexai=True,
    project=GCP_PROJECT_ID,
    location=GCP_LOCATION,
    http_options=types.HttpOptions(api_version="v1"),
)

# ==========================================
# [RAG] 만덕콜센터 안내 자료 검색
# ==========================================
# RAG_CORPUS 예: projects/385248657749/locations/asia-southeast1/ragCorpora/1234567890
# 비워두면 RAG를 건너뛰고 기존 Few-Shot만으로 동작한다 (켜고 끄는 스위치).
RAG_CORPUS = os.getenv("RAG_CORPUS", "").strip()
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_DISTANCE_THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.5"))

if RAG_CORPUS:
    # 코퍼스 리전은 Gemini 리전(GCP_LOCATION)과 다를 수 있다.
    # (RAG Engine은 신규 프로젝트에서 us-central1이 allowlist 전용)
    # → 리소스 이름 projects/.../locations/{리전}/ragCorpora/... 에서 리전을 꺼내 쓴다.
    _rag_parts = RAG_CORPUS.split("/")
    RAG_LOCATION = (
        _rag_parts[_rag_parts.index("locations") + 1]
        if "locations" in _rag_parts
        else GCP_LOCATION
    )
    agentplatform.init(project=GCP_PROJECT_ID, location=RAG_LOCATION)
    logger.info(
        "RAG 활성화: corpus=%s location=%s top_k=%s",
        RAG_CORPUS, RAG_LOCATION, RAG_TOP_K,
    )
else:
    logger.info("RAG 비활성화: RAG_CORPUS 미설정")


def retrieve_context(jeju_text: str) -> list[str]:
    """질문과 뜻이 가까운 안내 자료를 최대 RAG_TOP_K개 가져온다. 실패하면 빈 목록."""
    if not RAG_CORPUS or not jeju_text.strip():
        return []
    started = time.perf_counter()
    try:
        response = rag.retrieval_query(
            text=jeju_text,
            rag_resources=[rag.RagResource(rag_corpus=RAG_CORPUS)],
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=RAG_TOP_K,
                filter=rag.Filter(vector_distance_threshold=RAG_DISTANCE_THRESHOLD),
            ),
        )
    except Exception:
        logger.exception("RAG 검색 실패 → Few-Shot만으로 진행")
        return []

    contexts = list(response.contexts.contexts)
    logger.info(
        "RAG 검색 %d건 (%.0fms): %s",
        len(contexts),
        (time.perf_counter() - started) * 1000,
        [(c.source_display_name, round(c.score, 3)) for c in contexts],
    )
    return [c.text for c in contexts]

print("Jeju VITS TTS 모델 적재 중...")
tts_model = None
_tts_error = None
try:
    tts_model = JejuVITSEngine(
        config_path=TTS_CONFIG_PATH,
        checkpoint_path=TTS_CHECKPOINT_PATH,
        device=DEVICE,
    )
except Exception as exc:  # Keep server diagnosable before checkpoint is copied.
    _tts_error = str(exc)
    print(f"⚠️ TTS 비활성화: {_tts_error}")

print("✅ 모델 적재 단계 완료")


# ==========================================
# 3. Pipeline helpers
# ==========================================
def transcribe_jeju(audio_path: str) -> tuple[str, float]:
    speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
    inputs = processor(
        speech_array,
        sampling_rate=sampling_rate,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.inference_mode():
        output = stt_model.generate(
            **inputs,
            language="ko",
            task="transcribe",
            output_scores=True,
            return_dict_in_generate=True,
        )

    transcript = processor.batch_decode(
        output.sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # 데이터셋 저장 시 status(pending/approved) 판정에 쓰이는 STT confidence: 토큰별
    # 평균 로그확률을 exp()로 0~1 확률값으로 변환한다 (dataset_logger.save_training_sample 참고).
    transition_scores = stt_model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True
    )
    valid_scores = transition_scores[transition_scores > -1e9]
    confidence = torch.exp(valid_scores.mean()).item() if valid_scores.numel() > 0 else 0.0

    return transcript, confidence


def call_gemini_ars(
    jeju_text: str,
    conversation_history: list[ConversationTurn],
) -> tuple[GeminiARSResult, float]:
    response = gemini_client.models.generate_content(
        model=GEMINI_TUNED_ENDPOINT,
        contents=build_few_shot_contents(
            jeju_text,
            [turn.model_dump() for turn in conversation_history],
            references=retrieve_context(jeju_text),
        ),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.15,
            top_p=0.8,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json",
            response_schema=GeminiARSResult,
            response_logprobs=True,
        ),
    )

    # STT confidence(transcribe_jeju)와 동일한 원리: 응답 토큰의 평균 로그확률을
    # exp()로 0~1 확률값으로 변환. 값이 없으면(엔드포인트 미지원 등) 0.0으로 폴백.
    avg_logprobs = response.candidates[0].avg_logprobs if response.candidates else None
    translation_confidence = math.exp(avg_logprobs) if avg_logprobs is not None else 0.0

    if response.parsed is not None:
        if isinstance(response.parsed, GeminiARSResult):
            return response.parsed, translation_confidence
        return GeminiARSResult.model_validate(response.parsed), translation_confidence

    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    logger.warning(
        "Gemini structured 파싱 실패, raw text로 폴백 (finish_reason=%s, text_len=%s)",
        finish_reason,
        len(response.text) if response.text else 0,
    )

    if not response.text:
        raise RuntimeError("Gemini가 빈 응답을 반환했습니다.")
    return GeminiARSResult.model_validate_json(response.text), translation_confidence


def synthesize_ars_reply(text: str) -> bytes:
    if tts_model is None:
        raise RuntimeError(_tts_error or "TTS 모델이 초기화되지 않았습니다.")

    return tts_model.synthesize_wav(
        text,
        max_chars=int(os.getenv("TTS_MAX_CHARS", "45")),
        pause_ms=int(os.getenv("TTS_PAUSE_MS", "220")),
        tail_silence_ms=int(os.getenv("TTS_TAIL_SILENCE_MS", "350")),
        length_scale=float(os.getenv("TTS_LENGTH_SCALE", "1.10")),
        noise_scale=float(os.getenv("TTS_NOISE_SCALE", "0.667")),
        noise_scale_w=float(os.getenv("TTS_NOISE_SCALE_W", "0.35")),
    )


def parse_conversation_history(history_json: str) -> list[ConversationTurn]:
    """Validate the five most recent browser-owned demo turns.

    The API remains stateless for Cloud Run. The browser is the source of
    short-lived demo history and sends it with each multipart upload.
    """
    try:
        raw_history = json.loads(history_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail="history는 JSON 배열이어야 합니다.") from exc

    if not isinstance(raw_history, list):
        raise HTTPException(status_code=422, detail="history는 JSON 배열이어야 합니다.")
    if len(raw_history) > 5:
        raise HTTPException(status_code=422, detail="history는 최근 5턴까지만 보낼 수 있습니다.")

    try:
        return [ConversationTurn.model_validate(turn) for turn in raw_history]
    except Exception as exc:
        raise HTTPException(status_code=422, detail="history 항목 형식이 올바르지 않습니다.") from exc


# ==========================================
# 4. Endpoints
# ==========================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "stt_loaded": True,
        "gemini_model": GEMINI_TUNED_ENDPOINT,
        "tts_loaded": tts_model is not None,
        "tts_checkpoint": TTS_CHECKPOINT_PATH,
        "tts_error": _tts_error,
    }


@app.post("/translate")
async def translate_audio(
    file: UploadFile = File(...),
    history: str = Form("[]"),
):
    """Single-call AI ARS pipeline.

    Web audio -> Whisper Jeju STT -> tuned Gemini -> Jeju VITS -> JSON.

    WAV bytes are base64-encoded because a single HTTP body cannot normally be
    both a JSON document and a raw WAV file at the same time. Frontend can turn
    audio_base64 back into a Blob(audio/wav) and play it immediately.
    """
    conversation_history = parse_conversation_history(history)
    start_time = time.time()
    suffix = Path(file.filename or "input.wav").suffix or ".wav"

    audio_bytes = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        temp_file_path = tmp.name

    logger.info(
        "업로드 수신: filename=%s content_type=%s size=%d bytes",
        file.filename,
        file.content_type,
        len(audio_bytes),
    )

    try:
        # Model objects share CPU/GPU memory. Serialize demo requests to avoid
        # concurrent inference spikes on a single Cloud Run instance.
        async with INFERENCE_LOCK:
            jeju_text, stt_confidence = await asyncio.to_thread(transcribe_jeju, temp_file_path)
            if not jeju_text:
                raise HTTPException(status_code=422, detail="STT 결과가 비어 있습니다.")

            gemini_result, translation_confidence = await asyncio.to_thread(
                call_gemini_ars,
                jeju_text,
                conversation_history,
            )

            try:
                await asyncio.to_thread(
                    save_training_sample,
                    audio_path=temp_file_path,
                    jeju_text=jeju_text,
                    standard_text=gemini_result.standard_text,
                    stt_confidence=stt_confidence,
                    translation_confidence=translation_confidence,
                )
            except Exception:
                logger.warning("데이터셋 저장 호출 실패", exc_info=True)

            if tts_model is None:
                raise HTTPException(
                    status_code=503,
                    detail=f"TTS 모델이 준비되지 않았습니다: {_tts_error}",
                )

            wav_bytes = await asyncio.to_thread(
                synthesize_ars_reply,
                gemini_result.ars_reply_jeju,
            )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "AI ARS 처리 실패: file=%s temp_path=%s size=%s",
            file.filename,
            temp_file_path,
            os.path.getsize(temp_file_path) if os.path.exists(temp_file_path) else "N/A",
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"AI ARS 처리 실패: {exc}") from exc
    finally:
        try:
            os.remove(temp_file_path)
        except FileNotFoundError:
            pass

    end_time = time.time()
    return {
        "status": "success",
        "jeju_text": jeju_text,
        "standard_text": gemini_result.standard_text,
        "ars_reply_text": gemini_result.ars_reply_jeju,
        "audio_mime_type": "audio/wav",
        "audio_filename": "ars_reply.wav",
        "audio_sample_rate": tts_model.sample_rate if tts_model else 22050,
        "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
        "processing_time": round(end_time - start_time, 2),
    }


@app.post("/tts")
async def tts_only(request: TTSRequest):
    """Optional raw-WAV endpoint for debugging/frontend reuse."""
    if tts_model is None:
        raise HTTPException(status_code=503, detail=f"TTS 모델 미준비: {_tts_error}")
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="text가 비어 있습니다.")

    async with INFERENCE_LOCK:
        try:
            wav_bytes = await asyncio.to_thread(synthesize_ars_reply, request.text.strip())
        except Exception as exc:
            logger.error("TTS 생성 실패: text=%r", request.text, exc_info=True)
            raise HTTPException(status_code=500, detail=f"TTS 생성 실패: {exc}") from exc

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="jeju_tts.wav"'},
    )


@app.get("/dataset/stats")
async def dataset_stats():
    """Data-flywheel dashboard: counts per review status."""
    try:
        return await asyncio.to_thread(get_stats)
    except Exception as exc:
        logger.error("데이터셋 통계 조회 실패", exc_info=True)
        raise HTTPException(status_code=500, detail=f"데이터셋 통계 조회 실패: {exc}") from exc


@app.get("/dataset/samples")
async def dataset_samples(limit: int = 20, offset: int = 0):
    """Data-flywheel dashboard: one page of training samples, newest first."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    try:
        return await asyncio.to_thread(list_samples, limit=limit, offset=offset)
    except Exception as exc:
        logger.error(
            "데이터셋 샘플 목록 조회 실패: limit=%s offset=%s", limit, offset, exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"데이터셋 샘플 목록 조회 실패: {exc}") from exc


@app.get("/dataset/audio/{sample_id}")
async def dataset_audio(sample_id: str):
    """Data-flywheel dashboard: stream one training sample's audio from GCS."""
    try:
        audio_bytes = await asyncio.to_thread(get_audio_bytes, sample_id)
    except NotFound:
        raise HTTPException(status_code=404, detail="오디오를 찾을 수 없습니다.")
    except Exception as exc:
        logger.error("오디오 조회 실패: sample_id=%s", sample_id, exc_info=True)
        raise HTTPException(status_code=500, detail=f"오디오 조회 실패: {exc}") from exc
    return Response(content=audio_bytes, media_type="audio/wav")


class DatasetLabelUpdate(BaseModel):
    dialect_form: str | None = None
    standard_form: str | None = None
    status: Literal["approved", "rejected"]


@app.patch("/dataset/samples/{sample_id}")
async def update_dataset_sample(sample_id: str, payload: DatasetLabelUpdate):
    """Data-flywheel dashboard: record a human review decision (and optional
    label correction) for one sample. Only status and, optionally, the label
    text are updated in place."""
    try:
        updated = await asyncio.to_thread(
            update_sample_label,
            sample_id=sample_id,
            status=payload.status,
            dialect_form=payload.dialect_form,
            standard_form=payload.standard_form,
        )
    except NotFound:
        raise HTTPException(status_code=404, detail="데이터를 찾을 수 없습니다.")
    except Exception as exc:
        logger.error(
            "데이터셋 샘플 갱신 실패: sample_id=%s payload=%s",
            sample_id,
            payload.model_dump(),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"데이터셋 샘플 갱신 실패: {exc}") from exc
    return updated
