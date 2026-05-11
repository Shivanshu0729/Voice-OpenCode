import asyncio
import time
import re
import threading
from collections import deque
from typing import Callable, Awaitable

from livekit import rtc
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.agents.llm import ChatMessage

from agent.core.vad import build_vad
from agent.core.stt import build_stt, format_transcript
from agent.core.tts import build_tts, TTSController
from agent.core.intent import IntentClassifier, Intent
from agent.subagents.pool import SubagentPool, TaskType, SubagentResult
from agent.utils.logger import get_logger
from agent.utils.latency import LatencyTracker
from config.settings import settings
from agent.core.llm_adapter import GroqLLM

logger = get_logger(__name__)
_ENTRYPOINT_LOCK = threading.Lock()
_ENTRYPOINT_RUNNING = False


class _SafeSubscriptionFuture:
    def __init__(self, future: asyncio.Future[None]) -> None:
        self._future = future

    def set_result(self, result: None) -> None:
        if not self._future.done():
            self._future.set_result(result)

    def __getattr__(self, name: str):
        return getattr(self._future, name)

    def __await__(self):
        return self._future.__await__()


class OuterLoopAgent:
    def __init__(self, ctx: JobContext) -> None:
        self._ctx = ctx
        self._tracker = LatencyTracker()
        self._tts_controller: TTSController | None = None
        self._pool: SubagentPool | None = None
        self._classifier: IntentClassifier | None = None
        self._room: rtc.Room | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._agent_text_topic: str = "agent-chat"
        self._recent_spoken_texts: deque[str] = deque(maxlen=8)
        self._last_spoken_normalized: str = ""
        self._last_spoken_at: float = 0.0
        self._last_handled_transcript_normalized: str = ""
        self._last_handled_transcript_at: float = 0.0

    async def run(self) -> None:
        logger.info("outer_loop_starting")

        await self._ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
        room = self._ctx.room
        self._room = room
        logger.info("room_connected", room_name=room.name)

        for topic in ("", "chat", "lk.chat"):
            try:
                room.register_text_stream_handler(
                    topic,
                    lambda reader, sender_identity, topic=topic: asyncio.create_task(
                        self._on_text_stream(reader, sender_identity, topic)
                    ),
                )
            except ValueError:
                continue
        logger.info("text_chat_registered")

        try:
            room.register_byte_stream_handler(
                "lk.agent.session",
                lambda reader, sender_identity: asyncio.create_task(
                    self._on_agent_session_stream(reader, sender_identity)
                ),
            )
        except ValueError:
            pass

        vad = build_vad()
        stt = build_stt()
        tts = build_tts()

        self._classifier = IntentClassifier(self._tracker)
        self._tts_controller = TTSController(tts, self._tracker)

        self._audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1)
        audio_track = rtc.LocalAudioTrack.create_audio_track("agent-audio", self._audio_source)
        audio_publication = await room.local_participant.publish_track(audio_track)
        audio_publication._first_subscription = _SafeSubscriptionFuture(audio_publication._first_subscription)
        logger.info("audio_track_published")

        self._pool = SubagentPool(on_progress=self._on_subagent_progress)

        llm_adapter = GroqLLM()

        agent = VoicePipelineAgent(
            vad=vad,
            stt=stt,
            llm=llm_adapter,
            tts=tts,
            before_llm_cb=self._suppress_pipeline_reply,
        )

        agent.on(
            "agent_speech_interrupted",
            lambda: asyncio.create_task(self._on_tts_interrupted()),
        )
        agent.on(
            "user_speech_started",
            lambda: asyncio.create_task(self._on_speech_started()),
        )
        agent.on(
            "user_stopped_speaking",
            lambda: asyncio.create_task(self._on_user_stopped_speaking(agent)),
        )

        agent.start(room)

        human_input = getattr(agent, "_human_input", None)
        if human_input is not None:
            human_input.on(
                "final_transcript",
                lambda ev: asyncio.create_task(self._on_final_transcript(ev)),
            )

        await asyncio.sleep(0.5)
        await self._speak("Ready. What would you like me to build?")

        asyncio.create_task(self._result_narrator_loop())

        logger.info("outer_loop_ready")
        await asyncio.Event().wait()

    async def _on_text_stream(self, reader, sender_identity: str, topic: str) -> None:
        if self._room:
            local_identity = getattr(self._room.local_participant, "identity", None)
            if local_identity and sender_identity == local_identity:
                return
        text = await reader.read_all()
        if not text.strip():
            return

        logger.info("text_message_received", topic=topic, text=text[:120])
        await self._on_utterance(text, None)

    async def _on_agent_session_stream(self, reader, sender_identity: str) -> None:
        payload = bytearray()
        async for chunk in reader:
            payload.extend(chunk)

        if payload:
            logger.debug(
                "agent_session_stream_received",
                sender_identity=sender_identity,
                bytes=len(payload),
            )

    def _suppress_pipeline_reply(self, *_args, **_kwargs):
        return False

    async def _on_final_transcript(self, event) -> None:
        transcript = self._extract_transcript_text(event).strip()
        if not transcript or transcript == "None":
            return

        await self._handle_user_transcript(transcript)

    async def _on_user_stopped_speaking(self, agent: VoicePipelineAgent) -> None:
        transcript = ""
        for _ in range(25):
            transcript = self._coerce_text(getattr(agent, "_transcribed_text", "")).strip()
            if transcript and transcript != "None":
                break
            await asyncio.sleep(0.04)

        if not transcript or transcript == "None":
            return

        await self._handle_user_transcript(transcript)

    async def _handle_user_transcript(self, transcript: str) -> None:
        normalized = self._normalize_text(transcript)
        now = time.perf_counter()
        if normalized and normalized == self._last_handled_transcript_normalized and (now - self._last_handled_transcript_at) < 2.0:
            return

        self._last_handled_transcript_normalized = normalized
        self._last_handled_transcript_at = now

        if self._looks_like_echo(transcript):
            logger.info("echo_transcript_ignored", text=transcript[:120])
            return

        await self._on_utterance(transcript, None)

    def _extract_transcript_text(self, event) -> str:
        transcript = self._coerce_text(getattr(event, "text", "")).strip()
        if transcript and transcript != "None":
            return transcript

        alternatives = getattr(event, "alternatives", None)
        if alternatives:
            first = alternatives[0]
            transcript = self._coerce_text(getattr(first, "text", first)).strip()
            if transcript and transcript != "None":
                return transcript

        transcript = self._coerce_text(event).strip()
        if transcript == "None":
            return ""
        return transcript

    async def _on_utterance(self, transcript: str | ChatMessage, speaker_id: int | None = None) -> None:
        utterance_start = time.perf_counter()
        transcript_text = self._coerce_text(transcript)
        formatted = format_transcript(transcript_text, speaker_id)
        logger.info("utterance_committed", text=str(formatted)[:120])

        if self._looks_like_echo(transcript_text):
            logger.info("echo_transcript_ignored", text=str(formatted)[:120])
            return

        try:
            try:
                classified = await self._classifier.classify(transcript_text)
            except Exception as exc:
                logger.exception("intent_classification_failed", error=str(exc))
                classified = None

            if classified is None:
                await self._speak("I had trouble understanding that. Try again with a simpler request.")
                return

            elapsed = (time.perf_counter() - utterance_start) * 1000
            self._tracker.record("end_to_end", elapsed)

            await self._dispatch(classified)
        except Exception as exc:
            logger.exception("utterance_handling_failed", error=str(exc))
            await self._speak("I hit a problem handling that request. Please try again.")

    async def _on_speech_started(self) -> None:
        if self._tts_controller and self._tts_controller.is_speaking:
            logger.info("interruption_detected")
            self._tts_controller.cancel()
            if self._pool:
                cancelled = self._pool.cancel()
                if cancelled:
                    logger.info("subagents_cancelled_on_interrupt", count=cancelled)

    async def _on_tts_interrupted(self) -> None:
        logger.debug("tts_interrupted_livekit_event")

    async def _dispatch(self, classified) -> None:
        intent = classified.intent
        prompt = classified.task_summary
        file_hint = classified.target_file

        match intent:
            case Intent.STOP:
                cancelled = self._pool.cancel()
                if cancelled:
                    await self._speak(
                        f"Stopped. Cancelled {cancelled} task{'s' if cancelled > 1 else ''}."
                    )
                else:
                    await self._speak("Nothing running to stop.")

            case Intent.CODING_TASK:
                await self._speak(self._classifier.immediate_ack(intent))
                self._pool.dispatch(TaskType.CODING, prompt, file_hint)

            case Intent.BUG_FIX:
                await self._speak(self._classifier.immediate_ack(intent))
                self._pool.dispatch(TaskType.BUG_FIX, prompt, file_hint)

            case Intent.RUN_COMMAND:
                await self._speak(self._classifier.immediate_ack(intent))
                self._pool.dispatch(TaskType.RUN_COMMAND, prompt)

            case Intent.FILE_READ:
                target = file_hint or prompt
                await self._speak(self._classifier.immediate_ack(intent))
                self._pool.dispatch(TaskType.FILE_READ, target, file_hint)

            case Intent.QUESTION:
                await self._speak(self._classifier.immediate_ack(intent))
                self._pool.dispatch(TaskType.QUESTION, prompt)

            case Intent.UNKNOWN:
                await self._speak(
                    "I did not quite understand that. Try saying something like: "
                    "write a function to, fix the bug in, or run the tests."
                )

    async def _result_narrator_loop(self) -> None:
        logger.debug("result_narrator_started")
        while True:
            await asyncio.sleep(0.2)

            if not self._pool:
                continue

            results: list[SubagentResult] = await self._pool.drain_results()
            for result in results:
                if result.cancelled:
                    continue
                logger.info(
                    "narrating_result",
                    task_id=result.task_id,
                    success=result.success,
                    summary=result.summary[:80],
                )
                await self._speak(result.summary)

    async def _on_subagent_progress(self, task_id: str, message: str) -> None:
        significant_keywords = {
            "writing", "reading", "fixing", "running", "installing", "done", "error"
        }
        message_text = self._coerce_text(message)
        lower = message_text.lower()
        if any(kw in lower for kw in significant_keywords) and len(message_text) > 10:
            logger.debug("subagent_progress_narrated", task_id=task_id, msg=message_text[:60])

    async def _speak(self, text: str | ChatMessage) -> None:
        if not self._tts_controller or not self._audio_source:
            logger.warning("speak_called_before_ready")
            return

        text_value = self._coerce_text(text)
        normalized = self._normalize_text(text_value)
        now = time.perf_counter()
        if normalized and normalized == self._last_spoken_normalized and (now - self._last_spoken_at) < 3.0:
            logger.debug("duplicate_speech_suppressed", text=str(text_value)[:80])
            return

        logger.debug("speaking", text=str(text_value)[:80])
        self._last_spoken_normalized = normalized
        self._last_spoken_at = now
        self._recent_spoken_texts.append(normalized)

        if self._room:
            try:
                await self._room.local_participant.send_text(text_value, topic=self._agent_text_topic)
            except Exception as exc:
                logger.debug("chat_text_publish_failed", error=str(exc))

        await self._tts_controller.speak(text_value, self._audio_source)

    def _normalize_text(self, text: str | ChatMessage | object) -> str:
        text_value = self._coerce_text(text)
        cleaned = re.sub(r"[^a-z0-9\s]", " ", text_value.lower())
        return " ".join(cleaned.split())

    def _coerce_text(self, value: str | ChatMessage | object) -> str:
        if isinstance(value, ChatMessage):
            content = value.content
            if content is None:
                return ""
            if isinstance(content, list):
                parts = [self._coerce_text(item) for item in content if item is not None]
                return " ".join(part for part in parts if part)
            return str(content)
        if hasattr(value, "content"):
            content = getattr(value, "content")
            if content is None:
                return ""
            if isinstance(content, list):
                parts = [self._coerce_text(item) for item in content if item is not None]
                return " ".join(part for part in parts if part)
            return str(content)
        return str(value)

    def _looks_like_echo(self, transcript: str | ChatMessage | object) -> bool:
        transcript_text = self._coerce_text(transcript)
        normalized = self._normalize_text(transcript_text)
        if not normalized:
            return False

        for spoken in self._recent_spoken_texts:
            if not spoken:
                continue
            if normalized == spoken:
                return True
            if normalized in spoken or spoken in normalized:
                return True

        return False


async def entrypoint(ctx: JobContext) -> None:
    global _ENTRYPOINT_RUNNING

    with _ENTRYPOINT_LOCK:
        if _ENTRYPOINT_RUNNING:
            logger.warning("duplicate_entrypoint_ignored")
            return
        _ENTRYPOINT_RUNNING = True

    try:
        agent = OuterLoopAgent(ctx)
        await agent.run()
    finally:
        with _ENTRYPOINT_LOCK:
            _ENTRYPOINT_RUNNING = False