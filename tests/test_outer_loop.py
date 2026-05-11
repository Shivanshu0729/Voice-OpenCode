import asyncio
import pytest
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from livekit.agents.llm import ChatMessage

from agent.core.intent import Intent, IntentClassifier, ClassifiedIntent
from agent.core.outer_loop import OuterLoopAgent, _SafeSubscriptionFuture
from agent.subagents.pool import TaskType
from agent.utils.latency import LatencyTracker


class TestIntentClassifier:
    def test_immediate_ack_returns_string_for_all_intents(self):
        tracker = LatencyTracker()
        classifier = IntentClassifier(tracker)

        for intent in Intent:
            ack = classifier.immediate_ack(intent)
            assert isinstance(ack, str)
            assert len(ack) > 0

    def test_stop_intent_ack_contains_stop_keyword(self):
        tracker = LatencyTracker()
        classifier = IntentClassifier(tracker)
        ack = classifier.immediate_ack(Intent.STOP)
        assert "stop" in ack.lower() or "stopping" in ack.lower()

    @pytest.mark.asyncio
    async def test_classify_returns_correct_intent(self):
        tracker = LatencyTracker()
        classifier = IntentClassifier(tracker)

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='{"intent": "coding_task", "confidence": 0.92, "target_file": null, "task_summary": "Write a sorting function"}'))
        ]

        with patch.object(
            classifier._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ):
            result = await classifier.classify("write a sorting function in Python")

        assert result.intent == Intent.CODING_TASK
        assert result.confidence == pytest.approx(0.92)
        assert result.task_summary == "Write a sorting function"

    @pytest.mark.asyncio
    async def test_classify_degrades_gracefully_on_parse_error(self):
        tracker = LatencyTracker()
        classifier = IntentClassifier(tracker)

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="not valid json"))]

        with patch.object(
            classifier._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
        ):
            result = await classifier.classify("mumble mumble")

        assert result.intent == Intent.UNKNOWN
        assert result.confidence == 0.0


class TestLatencyTracker:
    def test_record_populates_samples(self):
        tracker = LatencyTracker()
        tracker.record("vad_detection", 85.0)
        tracker.record("vad_detection", 92.0)
        assert len(tracker._samples["vad_detection"]) == 2

    def test_report_does_not_raise(self):
        tracker = LatencyTracker()
        tracker.record("end_to_end", 1150.0)
        tracker.report()

    def test_span_context_manager_records_time(self):
        tracker = LatencyTracker()
        with tracker.span("test_stage"):
            pass
        assert "test_stage" in tracker._samples
        assert tracker._samples["test_stage"][0] >= 0.0


class TestOuterLoopAgentSpeechGuards:
    @pytest.mark.asyncio
    async def test_ignores_echo_transcript(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)
        agent._recent_spoken_texts = deque(["ready what would you like me to build"], maxlen=8)
        agent._last_spoken_normalized = "ready what would you like me to build"
        agent._last_spoken_at = 0.0
        agent._classifier = AsyncMock()
        agent._speak = AsyncMock()
        agent._tracker = LatencyTracker()

        await agent._on_utterance("Ready, what would you like me to build?")

        agent._classifier.classify.assert_not_awaited()
        agent._speak.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_falls_back_when_classify_raises(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)
        agent._recent_spoken_texts = deque(maxlen=8)
        agent._last_spoken_normalized = ""
        agent._last_spoken_at = 0.0
        agent._classifier = SimpleNamespace(
            classify=AsyncMock(side_effect=RuntimeError("boom")),
            immediate_ack=MagicMock(return_value="On it."),
        )
        agent._speak = AsyncMock()
        agent._tracker = LatencyTracker()
        agent._pool = None

        await agent._on_utterance("write a function")

        agent._speak.assert_awaited()
        assert "trouble understanding" in agent._speak.await_args_list[-1].args[0].lower()

    @pytest.mark.asyncio
    async def test_accepts_chat_message_transcript(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)
        agent._recent_spoken_texts = deque(maxlen=8)
        agent._last_spoken_normalized = ""
        agent._last_spoken_at = 0.0
        agent._classifier = SimpleNamespace(
            classify=AsyncMock(return_value=SimpleNamespace(intent=Intent.UNKNOWN, task_summary="", target_file=None)),
            immediate_ack=MagicMock(return_value="On it."),
        )
        agent._speak = AsyncMock()
        agent._tracker = LatencyTracker()
        agent._pool = SimpleNamespace(cancel=MagicMock(return_value=0), dispatch=MagicMock())

        await agent._on_utterance(ChatMessage.create(text="My voice is clear.", role="user"))

        agent._classifier.classify.assert_awaited_once()
        agent._speak.assert_awaited()

    def test_normalize_text_accepts_chat_message(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)

        normalized = agent._normalize_text(ChatMessage.create(text="Hello, World!", role="user"))

        assert normalized == "hello world"

    @pytest.mark.asyncio
    async def test_text_stream_routes_to_utterance(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)
        agent._room = SimpleNamespace(local_participant=SimpleNamespace(identity="agent"))
        agent._on_utterance = AsyncMock()

        reader = SimpleNamespace(read_all=AsyncMock(return_value="Write a function."))

        await agent._on_text_stream(reader, sender_identity="user", topic="chat")

        agent._on_utterance.assert_awaited_once_with("Write a function.", None)

    @pytest.mark.asyncio
    async def test_speak_publishes_agent_text_on_separate_topic(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)
        agent._tts_controller = SimpleNamespace(
            is_speaking=False,
            disabled=False,
            speak=AsyncMock(),
        )
        agent._audio_source = SimpleNamespace()
        agent._room = SimpleNamespace(
            local_participant=SimpleNamespace(send_text=AsyncMock())
        )
        agent._agent_text_topic = "agent-chat"
        agent._last_spoken_normalized = ""
        agent._last_spoken_at = 0.0
        agent._recent_spoken_texts = deque(maxlen=8)

        await agent._speak("Hello there")

        agent._room.local_participant.send_text.assert_awaited_once_with(
            "Hello there", topic="agent-chat"
        )

    @pytest.mark.asyncio
    async def test_speak_still_calls_tts_when_disabled(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)
        agent._tts_controller = SimpleNamespace(
            is_speaking=False,
            disabled=True,
            speak=AsyncMock(),
        )
        agent._audio_source = SimpleNamespace()
        agent._room = SimpleNamespace(
            local_participant=SimpleNamespace(send_text=AsyncMock())
        )
        agent._agent_text_topic = "agent-chat"
        agent._last_spoken_normalized = ""
        agent._last_spoken_at = 0.0
        agent._recent_spoken_texts = deque(maxlen=8)

        await agent._speak("Text only")

        agent._tts_controller.speak.assert_awaited_once_with("Text only", agent._audio_source)
        agent._room.local_participant.send_text.assert_awaited_once_with(
            "Text only", topic="agent-chat"
        )

    @pytest.mark.asyncio
    async def test_dispatch_speaks_immediate_ack_before_coding_task(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)
        agent._classifier = SimpleNamespace(immediate_ack=MagicMock(return_value="On it, writing that now."))
        agent._speak = AsyncMock()
        agent._pool = SimpleNamespace(dispatch=MagicMock())

        classified = SimpleNamespace(
            intent=Intent.CODING_TASK,
            task_summary="Write a sorting function",
            target_file=None,
        )

        await agent._dispatch(classified)

        agent._speak.assert_awaited_once_with("On it, writing that now.")
        agent._pool.dispatch.assert_called_once_with(
            TaskType.CODING, "Write a sorting function", None
        )

    def test_suppress_pipeline_reply_returns_false(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)

        assert agent._suppress_pipeline_reply(None, None) is False

    @pytest.mark.asyncio
    async def test_safe_subscription_future_ignores_duplicate_set_result(self):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        safe_future = _SafeSubscriptionFuture(future)

        safe_future.set_result(None)
        safe_future.set_result(None)

        await asyncio.shield(safe_future)
        assert future.done()

    @pytest.mark.asyncio
    async def test_agent_session_stream_consumer_reads_all_bytes(self):
        agent = OuterLoopAgent.__new__(OuterLoopAgent)

        class _Reader:
            def __init__(self):
                self._chunks = [b"hello", b" world"]

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._chunks:
                    raise StopAsyncIteration
                return self._chunks.pop(0)

        await agent._on_agent_session_stream(_Reader(), "shivanshu")