import audioop
import asyncio
import os
import tempfile
import wave
from contextlib import suppress

from livekit import rtc
from livekit.agents import APIConnectionError
from livekit.agents._exceptions import APIError
from livekit.plugins import cartesia
import pyttsx3

from config.settings import settings
from agent.utils.logger import get_logger
from agent.utils.latency import LatencyTracker

logger = get_logger(__name__)


def build_tts() -> cartesia.TTS:
    logger.info("initialising_cartesia_tts", voice_id=settings.cartesia_voice_id)
    return cartesia.TTS(
        voice=settings.cartesia_voice_id,
        model="sonic-2",
        encoding="pcm_s16le",
        sample_rate=24000,
        api_key=settings.cartesia_api_key,
    )


class TTSController:

    def __init__(self, tts: cartesia.TTS, tracker: LatencyTracker) -> None:
        self._tts = tts
        self._tracker = tracker
        self._current_task: asyncio.Task | None = None
        self._interrupted = False
        self._cartesia_available = True

    @property
    def is_speaking(self) -> bool:
        return self._current_task is not None and not self._current_task.done()

    def cancel(self) -> None:
        if self._current_task and not self._current_task.done():
            self._interrupted = True
            self._current_task.cancel()
            logger.debug("tts_interrupted")

    @property
    def disabled(self) -> bool:
        return not self._cartesia_available

    async def speak(self, text: str, audio_source: rtc.AudioSource) -> None:
        self._interrupted = False

        async def _stream() -> None:
            import time
            start = time.perf_counter()

            stream = None
            try:
                if self._cartesia_available:
                    stream = self._tts.synthesize(text)
                    audio_frame = await stream.collect()
                    if self._interrupted:
                        return

                    elapsed_ms = (time.perf_counter() - start) * 1000
                    self._tracker.record("tts_first_audio", elapsed_ms)
                    await audio_source.capture_frame(audio_frame)
                    return

                await self._speak_with_local_tts(text, audio_source)
            except (APIConnectionError, APIError, Exception) as exc:
                logger.warning("tts_synthesis_failed", error=str(exc))
                self._cartesia_available = False
                await self._speak_with_local_tts(text, audio_source)
            finally:
                if stream is not None:
                    await stream.aclose()

        self._current_task = asyncio.create_task(_stream())
        try:
            await self._current_task
        except asyncio.CancelledError:
            logger.debug("tts_task_cancelled")
        finally:
            self._current_task = None

    async def _speak_with_local_tts(self, text: str, audio_source: rtc.AudioSource) -> None:
        try:
            frames = await asyncio.to_thread(self._build_local_audio_frames, text)
            if not frames or self._interrupted:
                return

            import time
            start = time.perf_counter()
            for frame in frames:
                if self._interrupted:
                    return
                await audio_source.capture_frame(frame)

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._tracker.record("tts_first_audio", elapsed_ms)
            logger.info("local_tts_fallback_used")
        except Exception as exc:
            logger.warning("local_tts_failed", error=str(exc))

    def _build_local_audio_frames(self, text: str) -> list[rtc.AudioFrame]:
        temp_path: str | None = None
        try:
            fd, temp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)

            engine = pyttsx3.init()
            engine.setProperty("rate", 180)
            engine.save_to_file(text, temp_path)
            engine.runAndWait()

            with wave.open(temp_path, "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                sample_rate = wav_file.getframerate()
                raw_pcm = wav_file.readframes(wav_file.getnframes())

            if channels != 1:
                raw_pcm = audioop.tomono(raw_pcm, sample_width, 1, 1)
                channels = 1

            target_rate = 24000
            if sample_rate != target_rate:
                raw_pcm, _ = audioop.ratecv(raw_pcm, sample_width, channels, sample_rate, target_rate, None)
                sample_rate = target_rate

            bytes_per_sample = sample_width * channels
            frame_samples = max(1, int(sample_rate * 0.02))
            chunk_size = frame_samples * bytes_per_sample
            frames: list[rtc.AudioFrame] = []

            for offset in range(0, len(raw_pcm), chunk_size):
                chunk = raw_pcm[offset : offset + chunk_size]
                if not chunk:
                    continue
                samples_per_channel = len(chunk) // bytes_per_sample
                frames.append(
                    rtc.AudioFrame(
                        chunk,
                        sample_rate,
                        channels,
                        samples_per_channel,
                    )
                )

            return frames
        finally:
            if temp_path:
                with suppress(FileNotFoundError):
                    os.remove(temp_path)