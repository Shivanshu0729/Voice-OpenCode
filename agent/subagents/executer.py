import asyncio
from dataclasses import dataclass

from config.settings import settings
from agent.utils.logger import get_logger

logger = get_logger(__name__)

MAX_OUTPUT_CHARS = 8_000


def _workspace() -> str:
    try:
        return settings.opencode_workspace
    except Exception:
        return "."


def _subagent_timeout() -> int:
    try:
        return settings.subagent_timeout
    except Exception:
        return 120


@dataclass
class ExecutionResult:
    command: str
    stdout: str
    stderr: str
    return_code: int
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.return_code == 0 and not self.timed_out

    def to_summary(self) -> str:
        if self.timed_out:
            return f"Command timed out after {_subagent_timeout()} seconds."
        if self.success:
            lines = self.stdout.strip().splitlines()
            preview = "\n".join(lines[-10:]) if lines else "(no output)"
            return f"Done. Output:\n{preview}"
        else:
            error_preview = (self.stderr or self.stdout).strip()[-500:]
            return f"Command failed (exit {self.return_code}):\n{error_preview}"


class CommandExecutor:

    def __init__(self, on_progress=None) -> None:
        self._on_progress = on_progress

    async def run(self, command: str) -> ExecutionResult:
        logger.info("executor_running", command=command[:120])
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=_workspace(),
            )

            async def read_stream(stream: asyncio.StreamReader, store: list[str], label: str):
                async for raw in stream:
                    line = raw.decode("utf-8", errors="replace")
                    store.append(line)
                    logger.debug(f"exec_{label}", line=line.rstrip())
                    if self._on_progress and label == "stdout":
                        await self._on_progress(line.rstrip())

            stdout_task = asyncio.create_task(read_stream(proc.stdout, stdout_chunks, "stdout"))
            stderr_task = asyncio.create_task(read_stream(proc.stderr, stderr_chunks, "stderr"))

            try:
                await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task, proc.wait()),
                    timeout=_subagent_timeout(),
                )
                timed_out = False
                return_code = proc.returncode
            except asyncio.TimeoutError:
                proc.kill()
                stdout_task.cancel()
                stderr_task.cancel()
                timed_out = True
                return_code = -1
                logger.warning("executor_timeout", command=command[:80])

            stdout = _truncate("".join(stdout_chunks))
            stderr = _truncate("".join(stderr_chunks))

            return ExecutionResult(
                command=command,
                stdout=stdout,
                stderr=stderr,
                return_code=return_code or 0,
                timed_out=timed_out,
            )

        except Exception as exc:
            logger.exception("executor_error", error=str(exc))
            return ExecutionResult(
                command=command,
                stdout="",
                stderr=str(exc),
                return_code=1,
            )


def _truncate(text: str) -> str:
    if len(text) > MAX_OUTPUT_CHARS:
        return "...[truncated]\n" + text[-MAX_OUTPUT_CHARS:]
    return text