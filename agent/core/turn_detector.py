import asyncio
import enum
import time

from livekit.plugins import turn_detector as lk_turn_detector

from agent.utils.logger import get_logger
from agent.utils.latency import LatencyTracker

logger = get_logger(__name__)

MAX_SILENCE_MS = 800


class TurnSignal(enum.Enum):
    END_OF_TURN = "end_of_turn"
    LIKELY_DONE = "likely_done"


class TurnDetector:

    def __init__(self, tracker: LatencyTracker) -> None:
        self._tracker = tracker
        self._model = lk_turn_detector.EOUModel()
        self._silence_start: float | None = None

    def on_speech_ended(self) -> None:
        self._silence_start = time.perf_counter()

    def on_speech_started(self) -> None:
        self._silence_start = None

    async def wait_for_turn_end(
        self,
        transcript: str,
        vad_silence_elapsed_ms: float,
    ) -> TurnSignal:
        start = time.perf_counter()

        score = await asyncio.get_event_loop().run_in_executor(
            None, self._model.predict, transcript
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._tracker.record("turn_detection", elapsed_ms)

        logger.debug(
            "turn_score",
            score=round(score, 3),
            silence_ms=round(vad_silence_elapsed_ms, 1),
        )

        if score >= 0.75:
            return TurnSignal.END_OF_TURN

        if score >= 0.50:
            return TurnSignal.LIKELY_DONE

        if vad_silence_elapsed_ms >= MAX_SILENCE_MS:
            logger.debug("turn_detection_fallback_fired", silence_ms=vad_silence_elapsed_ms)
            return TurnSignal.END_OF_TURN

        return TurnSignal.LIKELY_DONE