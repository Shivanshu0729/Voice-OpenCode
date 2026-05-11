from __future__ import annotations

import time
from dataclasses import dataclass

from livekit import rtc
from livekit.agents import vad as agents_vad

from agent.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class _SimpleVADConfig:
    threshold: float = 0.65
    min_speech_duration: float = 0.1
    min_silence_duration: float = 0.3
    update_interval: float = 0.2


class SimpleEnergyVADStream(agents_vad.VADStream):
    def __init__(self, vad: "SimpleEnergyVAD") -> None:
        super().__init__(vad)
        self._config = vad._config
        self._speaking = False
        self._speech_frames: list[rtc.AudioFrame] = []
        self._speech_duration = 0.0
        self._silence_duration = 0.0
        self._samples_index = 0

    @staticmethod
    def _frame_probability(frame: rtc.AudioFrame) -> float:
        samples = frame.data
        if len(samples) == 0:
            return 0.0

        mean_abs = sum(abs(int(sample)) for sample in samples) / len(samples)
        probability = mean_abs / 5000.0
        if probability < 0.0:
            return 0.0
        if probability > 1.0:
            return 1.0
        return probability

    async def _emit(self, event: agents_vad.VADEvent) -> None:
        await self._event_ch.send(event)

    async def _main_task(self) -> None:
        import time as _time

        start_time = _time.perf_counter()

        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                break

            frame = item
            frame_duration = frame.duration
            probability = self._frame_probability(frame)
            is_speech = probability >= self._config.threshold
            now = _time.perf_counter()

            if is_speech:
                self._silence_duration = 0.0
                self._speech_duration += frame_duration
                self._speech_frames.append(frame)

                if not self._speaking and self._speech_duration >= self._config.min_speech_duration:
                    self._speaking = True
                    await self._emit(
                        agents_vad.VADEvent(
                            type=agents_vad.VADEventType.START_OF_SPEECH,
                            samples_index=self._samples_index,
                            timestamp=now - start_time,
                            speech_duration=self._speech_duration,
                            silence_duration=self._silence_duration,
                            frames=list(self._speech_frames),
                            probability=probability,
                            inference_duration=0.0,
                            speaking=True,
                            raw_accumulated_silence=self._silence_duration,
                            raw_accumulated_speech=self._speech_duration,
                        )
                    )
            else:
                self._silence_duration += frame_duration
                if self._speaking and self._silence_duration >= self._config.min_silence_duration:
                    self._speaking = False
                    await self._emit(
                        agents_vad.VADEvent(
                            type=agents_vad.VADEventType.END_OF_SPEECH,
                            samples_index=self._samples_index,
                            timestamp=now - start_time,
                            speech_duration=self._speech_duration,
                            silence_duration=self._silence_duration,
                            frames=list(self._speech_frames),
                            probability=probability,
                            inference_duration=0.0,
                            speaking=False,
                            raw_accumulated_silence=self._silence_duration,
                            raw_accumulated_speech=self._speech_duration,
                        )
                    )
                    self._speech_frames.clear()
                    self._speech_duration = 0.0
                    self._silence_duration = 0.0

            await self._emit(
                agents_vad.VADEvent(
                    type=agents_vad.VADEventType.INFERENCE_DONE,
                    samples_index=self._samples_index,
                    timestamp=now - start_time,
                    speech_duration=self._speech_duration,
                    silence_duration=self._silence_duration,
                    frames=[frame],
                    probability=probability,
                    inference_duration=0.0,
                    speaking=self._speaking,
                    raw_accumulated_silence=self._silence_duration,
                    raw_accumulated_speech=self._speech_duration,
                )
            )

            self._samples_index += frame.samples_per_channel

        if self._speaking and self._speech_frames:
            now = _time.perf_counter()
            await self._emit(
                agents_vad.VADEvent(
                    type=agents_vad.VADEventType.END_OF_SPEECH,
                    samples_index=self._samples_index,
                    timestamp=now - start_time,
                    speech_duration=self._speech_duration,
                    silence_duration=self._silence_duration,
                    frames=list(self._speech_frames),
                    probability=0.0,
                    inference_duration=0.0,
                    speaking=False,
                    raw_accumulated_silence=self._silence_duration,
                    raw_accumulated_speech=self._speech_duration,
                )
            )
            self._speech_frames.clear()


class SimpleEnergyVAD(agents_vad.VAD):
    def __init__(self, *, threshold: float = 0.65) -> None:
        self._config = _SimpleVADConfig(threshold=threshold)
        super().__init__(capabilities=agents_vad.VADCapabilities(update_interval=self._config.update_interval))

    def stream(self) -> agents_vad.VADStream:
        return SimpleEnergyVADStream(self)


def build_vad() -> agents_vad.VAD:
    logger.info("initialising_silero_vad", threshold=0.65)
    return SimpleEnergyVAD(threshold=0.65)
