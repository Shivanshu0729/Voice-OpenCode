from livekit.plugins import deepgram

from config.settings import settings
from agent.utils.logger import get_logger

logger = get_logger(__name__)


def build_stt() -> deepgram.STT:
    logger.info(
        "initialising_deepgram_stt",
        model=settings.deepgram_model,
        diarize=settings.deepgram_diarize,
    )
    return deepgram.STT(
        model=settings.deepgram_model,
        language="en-US",
        smart_format=True,
        interim_results=True,
        endpointing_ms=300,
        punctuate=True,
        api_key=settings.deepgram_api_key,
    )


def format_transcript(text: str, speaker_id: int | None) -> str:
    if speaker_id is not None and settings.deepgram_diarize:
        return f"[Speaker {speaker_id}] {text}"
    return text