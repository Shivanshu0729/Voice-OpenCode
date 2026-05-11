import os
from pathlib import Path
from types import SimpleNamespace

from openai import AsyncOpenAI

from config.settings import settings
from agent.utils.logger import get_logger

logger = get_logger(__name__)


def _groq_api_key() -> str:
    try:
        return settings.groq_api_key
    except Exception as exc:
        raise RuntimeError("Groq client is not configured") from exc


def _groq_model() -> str:
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

SUMMARISE_THRESHOLD_LINES = 50
ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".java",
    ".cpp", ".c", ".h", ".rb", ".sh", ".yaml", ".yml", ".toml",
    ".json", ".md", ".txt", ".sql",
}


class FileSystemAgent:

    def __init__(self) -> None:
        self._client = _LazyGroqClient(
            lambda: AsyncOpenAI(
                api_key=_groq_api_key(),
                base_url="https://api.groq.com/openai/v1",
            )
        )

    def _workspace(self) -> Path:
        try:
            return Path(settings.opencode_workspace)
        except Exception:
            return Path(".")

    def _resolve(self, filename: str) -> Path | None:
        workspace = self._workspace()
        target = workspace / filename
        if target.exists():
            return target

        for match in workspace.rglob(filename):
            if match.is_file():
                return match

        return None

    async def read_and_describe(self, filename: str) -> str:
        path = self._resolve(filename)
        if path is None:
            return f"I could not find a file called {filename} in your workspace."

        if path.suffix not in ALLOWED_EXTENSIONS:
            return f"{filename} is a binary or unsupported file type."

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except PermissionError:
            return f"I do not have permission to read {filename}."

        lines = content.splitlines()
        line_count = len(lines)

        if line_count <= SUMMARISE_THRESHOLD_LINES:
            return await self._describe_short(filename, content, line_count)
        else:
            return await self._summarise_long(filename, content, line_count)

    async def _describe_short(self, filename: str, content: str, line_count: int) -> str:
        response = await self._client.chat.completions.create(
            model=_groq_model(),
            max_tokens=300,
            messages=[
                {"role": "system", "content": "You are a concise voice-first coding assistant."},
                {
                    "role": "user",
                    "content": (
                        f"Describe this {line_count}-line file in 2-3 sentences for a voice interface. "
                        f"Be concise and specific. File: {filename}\n\n{content}"
                    ),
                },
            ],
        )
        return _extract_chat_text(response)

    async def _summarise_long(self, filename: str, content: str, line_count: int) -> str:
        lines = content.splitlines()
        head = "\n".join(lines[:200])
        tail = "\n".join(lines[-20:]) if line_count > 220 else ""
        excerpt = head + ("\n...\n" + tail if tail else "")

        response = await self._client.chat.completions.create(
            model=_groq_model(),
            max_tokens=400,
            messages=[
                {"role": "system", "content": "You are a concise voice-first coding assistant."},
                {
                    "role": "user",
                    "content": (
                        f"{filename} is {line_count} lines. Summarise its purpose and key components "
                        f"in 3-4 sentences for a voice assistant. Be specific.\n\n{excerpt}"
                    ),
                },
            ],
        )
        return _extract_chat_text(response)

    def list_workspace(self) -> str:
        try:
            workspace = self._workspace()
            items = sorted(workspace.iterdir(), key=lambda p: (p.is_file(), p.name))
            dirs = [p.name for p in items if p.is_dir() and not p.name.startswith(".")]
            files = [p.name for p in items if p.is_file() and p.suffix in ALLOWED_EXTENSIONS]
            parts = []
            if dirs:
                parts.append(f"Directories: {', '.join(dirs[:8])}")
            if files:
                parts.append(f"Files: {', '.join(files[:12])}")
            return ". ".join(parts) or "The workspace appears to be empty."
        except Exception as exc:
            logger.warning("list_workspace_error", error=str(exc))
            return "I could not list the workspace contents."