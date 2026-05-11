import json
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

from openai import AsyncOpenAI

from config.settings import settings
from agent.utils.logger import get_logger
from agent.utils.latency import LatencyTracker

logger = get_logger(__name__)


def _groq_api_key() -> str:
    try:
        return settings.groq_api_key
    except Exception as exc:
        raise RuntimeError("Groq client is not configured") from exc


def _outer_loop_model() -> str:
    try:
        return settings.outer_loop_model
    except Exception:
        return "llama-3.1-8b-instant"


def _extract_chat_text(response) -> str:
    try:
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise RuntimeError("Unexpected Groq response shape") from exc


class _LazyGroqCompletions:
    def __init__(self, client_factory):
        self._client_factory = client_factory

    async def create(self, *args, **kwargs):
        try:
            client = self._client_factory()
        except Exception as exc:
            raise RuntimeError("Groq client is not configured") from exc
        return await client.chat.completions.create(*args, **kwargs)


class _LazyGroqClient:
    def __init__(self, client_factory):
        self._client_factory = client_factory
        self.chat = SimpleNamespace(completions=_LazyGroqCompletions(client_factory))

    def _build(self):
        return self._client_factory()


class Intent(Enum):
    CODING_TASK = "coding_task"
    BUG_FIX = "bug_fix"
    RUN_COMMAND = "run_command"
    FILE_READ = "file_read"
    STOP = "stop"
    QUESTION = "question"
    UNKNOWN = "unknown"


@dataclass
class ClassifiedIntent:
    intent: Intent
    confidence: float
    target_file: str | None
    task_summary: str


_SYSTEM_PROMPT = """You are the intent router for a voice-controlled coding assistant.
Classify the user utterance and extract key entities.

Respond ONLY with a JSON object, no preamble, no explanation:
{
  "intent": "<coding_task | bug_fix | run_command | file_read | stop | question | unknown>",
  "confidence": <float 0.0-1.0>,
  "target_file": "<filename or null>",
  "task_summary": "<one sentence, imperative mood, specific>"
}

Intent definitions:
- coding_task : write, create, add, implement, generate new code
- bug_fix     : fix, debug, resolve an error or bug in existing code
- run_command : run, execute, test, install, build
- file_read   : show, read, open, display an existing file
- stop        : stop, cancel, halt, pause, never mind, wait
- question    : explain, what is, how does, why does
- unknown     : unclear or ambiguous request"""


class IntentClassifier:
    def __init__(self, tracker: LatencyTracker) -> None:
        self._client = _LazyGroqClient(
            lambda: AsyncOpenAI(
                api_key=_groq_api_key(),
                base_url="https://api.groq.com/openai/v1",
            )
        )
        self._tracker = tracker

    async def classify(self, transcript: str) -> ClassifiedIntent:
        import time
        start = time.perf_counter()
        raw = ""

        transcript_text = transcript
        if hasattr(transcript, 'content'):
            transcript_text = transcript.content
        elif not isinstance(transcript, str):
            transcript_text = str(transcript)

        try:
            response = await self._client.chat.completions.create(
                model=_outer_loop_model(),
                max_tokens=256,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": transcript_text},
                ],
            )
            raw = _extract_chat_text(response)
            data = json.loads(raw)

            elapsed_ms = (time.perf_counter() - start) * 1000
            self._tracker.record("llm_first_token", elapsed_ms)

            intent = Intent(data.get("intent", "unknown"))
            result = ClassifiedIntent(
                intent=intent,
                confidence=float(data.get("confidence", 0.5)),
                target_file=data.get("target_file"),
                task_summary=data.get("task_summary", transcript),
            )

            logger.info(
                "intent_classified",
                intent=intent.value,
                confidence=result.confidence,
                target_file=result.target_file,
            )
            return result

        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("intent_parse_failed", error=str(exc), raw=raw)
            return ClassifiedIntent(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                target_file=None,
                task_summary=transcript,
            )
        except Exception as exc:
            logger.warning("intent_classification_failed", error=str(exc), raw=raw)
            return ClassifiedIntent(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                target_file=None,
                task_summary=transcript,
            )

    def immediate_ack(self, intent: Intent) -> str:
        acks = {
            Intent.CODING_TASK: "On it, writing that now.",
            Intent.BUG_FIX: "Got it, looking at that bug.",
            Intent.RUN_COMMAND: "Running that for you.",
            Intent.FILE_READ: "Reading the file.",
            Intent.STOP: "Stopping.",
            Intent.QUESTION: "Sure, let me explain.",
            Intent.UNKNOWN: "Say that again, I did not quite catch what you need.",
        }
        return acks.get(intent, "On it.")