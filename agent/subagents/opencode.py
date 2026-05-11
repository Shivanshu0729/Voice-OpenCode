import asyncio
import json
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from typing import AsyncIterator

from config.settings import settings
from agent.utils.logger import get_logger

logger = get_logger(__name__)


def _opencode_binary() -> str:
    try:
        return settings.opencode_binary
    except Exception:
        return "opencode"


class EventType(Enum):
    PROGRESS = "progress"
    FILE_CHANGED = "file_changed"
    RESULT = "result"
    ERROR = "error"
    RAW = "raw"


@dataclass
class OpenCodeEvent:
    type: EventType
    message: str
    path: str | None = None
    line: int | None = None
    success: bool | None = None


class OpenCodeSession:

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None

    @staticmethod
    def is_available() -> bool:
        return shutil.which(_opencode_binary()) is not None

    async def run(self, prompt: str) -> AsyncIterator[OpenCodeEvent]:
        binary = _opencode_binary()

        if not self.is_available():
            logger.error("opencode_not_found", binary=binary)
            yield OpenCodeEvent(
                type=EventType.ERROR,
                message=(
                    f"OpenCode binary '{binary}' not found. "
                    "Install it from github.com/opencode-ai/opencode"
                ),
            )
            return

        cmd = [
            binary,
            "run",
            "--output", "json",
            "--no-tty",
            "--prompt", prompt,
        ]

        logger.info("opencode_starting", prompt=prompt[:80])

        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=settings.opencode_workspace,
                env=os.environ.copy(),
            )

            async for raw_line in self._process.stdout:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue

                event = self._parse_line(line)
                logger.debug("opencode_event", type=event.type.value, msg=event.message[:120])
                yield event

                if event.type in (EventType.RESULT, EventType.ERROR):
                    break

            _, stderr_bytes = await self._process.communicate()
            if stderr_bytes:
                logger.warning("opencode_stderr", output=stderr_bytes.decode()[:500])

        except asyncio.CancelledError:
            await self._terminate()
            raise
        except Exception as exc:
            logger.exception("opencode_unexpected_error", error=str(exc))
            yield OpenCodeEvent(type=EventType.ERROR, message=f"Unexpected error: {exc}")
        finally:
            self._process = None

    def _parse_line(self, line: str) -> OpenCodeEvent:
        try:
            data = json.loads(line)
            event_type = EventType(data.get("type", "raw"))
            return OpenCodeEvent(
                type=event_type,
                message=data.get("message") or data.get("summary", line),
                path=data.get("path"),
                line=data.get("line"),
                success=data.get("success"),
            )
        except (json.JSONDecodeError, ValueError):
            return OpenCodeEvent(type=EventType.RAW, message=line)

    async def _terminate(self) -> None:
        if self._process and self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
            logger.info("opencode_terminated")