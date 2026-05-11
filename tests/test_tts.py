import pytest
from unittest.mock import AsyncMock

from livekit.agents import APIConnectionError

from agent.core.tts import TTSController
from agent.utils.latency import LatencyTracker


class _FailingStream:
    async def collect(self):
        raise APIConnectionError("synthetic connection failure")

    async def aclose(self):
        return None


class _FailingTTS:
    def synthesize(self, text: str):
        return _FailingStream()


class _AudioSource:
    def __init__(self):
        self.frames = []

    async def capture_frame(self, frame):
        self.frames.append(frame)


@pytest.mark.asyncio
async def test_speak_swallow_connection_errors():
    controller = TTSController(_FailingTTS(), LatencyTracker())
    audio_source = _AudioSource()

    controller._build_local_audio_frames = lambda text: []

    await controller.speak("hello", audio_source)

    assert controller.is_speaking is False
    assert audio_source.frames == []


@pytest.mark.asyncio
async def test_speak_uses_local_fallback_when_cartesia_fails():
    controller = TTSController(_FailingTTS(), LatencyTracker())
    audio_source = _AudioSource()
    fake_frame = object()

    controller._build_local_audio_frames = lambda text: [fake_frame]

    await controller.speak("hello", audio_source)

    assert controller.disabled is True
    assert audio_source.frames == [fake_frame]