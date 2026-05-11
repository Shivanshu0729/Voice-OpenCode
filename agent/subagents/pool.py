import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Awaitable

from openai import AsyncOpenAI

from agent.subagents.opencode import OpenCodeSession, EventType
from agent.subagents.executer import CommandExecutor
from agent.subagents.filesystem import FileSystemAgent
from agent.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


def _groq_api_key() -> str:
    try:
        return settings.groq_api_key
    except Exception as exc:
        raise RuntimeError("Groq client is not configured") from exc


def _groq_model(name: str) -> str:
    try:
        return getattr(settings, name)
    except Exception:
        return "llama-3.1-8b-instant"


def _extract_chat_text(response) -> str:
    try:
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        raise RuntimeError("Unexpected Groq response shape") from exc


class TaskType(Enum):
    CODING = "coding"
    BUG_FIX = "bug_fix"
    RUN_COMMAND = "run_command"
    FILE_READ = "file_read"
    QUESTION = "question"


@dataclass
class SubagentResult:
    task_id: str
    task_type: TaskType
    success: bool
    summary: str
    details: str = ""
    cancelled: bool = False


@dataclass
class _RunningTask:
    task_id: str
    task_type: TaskType
    asyncio_task: asyncio.Task
    progress_msgs: list[str] = field(default_factory=list)


class SubagentPool:
    def __init__(self, on_progress: Callable[[str, str], Awaitable[None]] | None = None) -> None:
        self._running: dict[str, _RunningTask] = {}
        self._results: asyncio.Queue[SubagentResult] = asyncio.Queue()
        self._on_progress = on_progress
        self._fs_agent = FileSystemAgent()

    def dispatch(
        self, task_type: TaskType, prompt: str, file_hint: str | None = None
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        logger.info(
            "subagent_dispatching",
            task_id=task_id,
            type=task_type.value,
            prompt=prompt[:60],
        )

        coro = self._run_task(task_id, task_type, prompt, file_hint)
        asyncio_task = asyncio.create_task(coro, name=f"subagent-{task_id}")

        self._running[task_id] = _RunningTask(
            task_id=task_id,
            task_type=task_type,
            asyncio_task=asyncio_task,
        )
        return task_id

    def cancel(self, task_id: str | None = None) -> int:
        if task_id:
            task = self._running.pop(task_id, None)
            if task:
                task.asyncio_task.cancel()
                logger.info("subagent_cancelled", task_id=task_id)
                return 1
            return 0
        else:
            cancelled = 0
            for t in list(self._running.values()):
                t.asyncio_task.cancel()
                cancelled += 1
            self._running.clear()
            logger.info("subagent_all_cancelled", count=cancelled)
            return cancelled

    async def drain_results(self) -> list[SubagentResult]:
        results = []
        while not self._results.empty():
            try:
                results.append(self._results.get_nowait())
            except asyncio.QueueEmpty:
                break
        return results

    @property
    def active_count(self) -> int:
        return len(self._running)

    async def _run_task(
        self,
        task_id: str,
        task_type: TaskType,
        prompt: str,
        file_hint: str | None,
    ) -> None:
        result = None
        try:
            match task_type:
                case TaskType.CODING | TaskType.BUG_FIX:
                    result = await self._run_opencode(task_id, task_type, prompt)
                case TaskType.RUN_COMMAND:
                    result = await self._run_command(task_id, prompt)
                case TaskType.FILE_READ:
                    result = await self._run_file_read(task_id, file_hint or prompt)
                case TaskType.QUESTION:
                    result = await self._run_question(task_id, prompt)
                case _:
                    result = SubagentResult(
                        task_id=task_id,
                        task_type=task_type,
                        success=False,
                        summary="I am not sure how to handle that request.",
                    )
        except asyncio.CancelledError:
            result = SubagentResult(
                task_id=task_id,
                task_type=task_type,
                success=False,
                summary="Task cancelled.",
                cancelled=True,
            )
        except Exception as exc:
            logger.exception("subagent_error", task_id=task_id, error=str(exc))
            result = SubagentResult(
                task_id=task_id,
                task_type=task_type,
                success=False,
                summary="I had trouble doing that just now. Please try again.",
            )
        finally:
            self._running.pop(task_id, None)
            if result:
                await self._results.put(result)

    async def _run_opencode(
        self, task_id: str, task_type: TaskType, prompt: str
    ) -> SubagentResult:
        session = OpenCodeSession()
        summary_parts = []
        last_progress = ""

        async def _stream():
            nonlocal last_progress
            async for event in session.run(prompt):
                if event.type in (EventType.PROGRESS, EventType.RAW) and event.message:
                    last_progress = event.message
                    if self._on_progress:
                        await self._on_progress(task_id, event.message)
                elif event.type == EventType.FILE_CHANGED:
                    msg = f"Modified {event.path}" + (
                        f" at line {event.line}" if event.line else ""
                    )
                    summary_parts.append(msg)
                    if self._on_progress:
                        await self._on_progress(task_id, msg)
                elif event.type == EventType.RESULT:
                    summary_parts.append(event.message)
                elif event.type == EventType.ERROR:
                    raise RuntimeError(event.message)

        await asyncio.wait_for(_stream(), timeout=settings.subagent_timeout)

        summary = " ".join(summary_parts) if summary_parts else (last_progress or "Done.")
        return SubagentResult(
            task_id=task_id, task_type=task_type, success=True, summary=summary
        )

    async def _run_command(self, task_id: str, prompt: str) -> SubagentResult:
        client = AsyncOpenAI(
            api_key=_groq_api_key(),
            base_url="https://api.groq.com/openai/v1",
        )
        resp = await client.chat.completions.create(
            model=_groq_model("outer_loop_model"),
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": "You convert natural language into a shell command and reply with only the command.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Convert this voice request to a shell command. "
                        f"Reply with ONLY the command, nothing else.\n\nRequest: {prompt}"
                    ),
                },
            ],
        )
        command = _extract_chat_text(resp).strip("`")

        async def _on_line(line: str):
            if self._on_progress:
                await self._on_progress(task_id, line)

        executor = CommandExecutor(on_progress=_on_line)
        exec_result = await executor.run(command)

        return SubagentResult(
            task_id=task_id,
            task_type=TaskType.RUN_COMMAND,
            success=exec_result.success,
            summary=exec_result.to_summary(),
            details=exec_result.stdout + exec_result.stderr,
        )

    async def _run_file_read(self, task_id: str, filename: str) -> SubagentResult:
        description = await self._fs_agent.read_and_describe(filename)
        return SubagentResult(
            task_id=task_id,
            task_type=TaskType.FILE_READ,
            success=True,
            summary=description,
        )

    async def _run_question(self, task_id: str, prompt: str) -> SubagentResult:
        try:
            client = AsyncOpenAI(
                api_key=_groq_api_key(),
                base_url="https://api.groq.com/openai/v1",
            )
            resp = await client.chat.completions.create(
                model=_groq_model("subagent_model"),
                max_tokens=500,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a senior software engineer answering questions verbally. "
                            "Keep answers concise and under 4 sentences as they will be read aloud."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            return SubagentResult(
                task_id=task_id,
                task_type=TaskType.QUESTION,
                success=True,
                summary=_extract_chat_text(resp),
            )
        except Exception as exc:
            logger.warning("question_answer_failed", task_id=task_id, error=str(exc))
            return SubagentResult(
                task_id=task_id,
                task_type=TaskType.QUESTION,
                success=False,
                summary="I could not get an answer right now. Please try again.",
            )