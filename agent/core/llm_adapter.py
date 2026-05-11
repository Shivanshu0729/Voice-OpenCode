from __future__ import annotations

import uuid
from typing import Any
from dotenv import load_dotenv
from openai import AsyncOpenAI
from livekit.agents._exceptions import APIConnectionError
from livekit.agents.llm import ChatChunk, ChatContext, Choice, ChoiceDelta, LLM, LLMCapabilities, LLMStream
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS
from config.settings import settings

load_dotenv()

def _groq_client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=settings.groq_api_key,
        base_url="https://api.groq.com/openai/v1",
    )

def _chat_messages(chat_ctx: ChatContext) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in chat_ctx.messages:
        content = message.content
        if content is None:
            continue
        if isinstance(content, list):
            content = " ".join(str(item) for item in content if item is not None)
        messages.append({"role": message.role, "content": str(content)})
    if not messages:
        messages.append(
            {
                "role": "system",
                "content": "You are a concise voice assistant for a coding workspace.",
            }
        )
    return messages


class GroqLLMStream(LLMStream):
    def __init__(
        self,
        llm: "GroqLLM",
        *,
        chat_ctx: ChatContext,
        fnc_ctx,
        conn_options: APIConnectOptions,
        temperature: float | None,
        n: int | None,
        parallel_tool_calls: bool | None,
        tool_choice,
    ) -> None:
        super().__init__(llm, chat_ctx=chat_ctx, fnc_ctx=fnc_ctx, conn_options=conn_options)
        self._temperature = temperature
        self._n = n
        self._parallel_tool_calls = parallel_tool_calls
        self._tool_choice = tool_choice

    async def _run(self) -> None:
        try:
            response = await _groq_client().chat.completions.create(
                model=settings.outer_loop_model,
                messages=_chat_messages(self.chat_ctx),
                temperature=self._temperature if self._temperature is not None else 0.4,
                max_tokens=256,
                n=self._n or 1,
            )
            text = (response.choices[0].message.content or "").strip()
            if not text:
                text = "I am here."

            self._event_ch.send_nowait(
                ChatChunk(
                    request_id=uuid.uuid4().hex,
                    choices=[Choice(delta=ChoiceDelta(role="assistant", content=text))],
                )
            )
        except Exception as exc:
            message = str(exc).lower()
            if "rate limit" in message or "rate_limit_exceeded" in message or "429" in message:
                fallback = "I am temporarily rate-limited right now, but I am still here."
                self._event_ch.send_nowait(
                    ChatChunk(
                        request_id=uuid.uuid4().hex,
                        choices=[Choice(delta=ChoiceDelta(role="assistant", content=fallback))],
                    )
                )
                return
            raise APIConnectionError(f"Groq completion failed: {exc}") from exc


class GroqLLM(LLM):
    def __init__(self) -> None:
        super().__init__(capabilities=LLMCapabilities())

    def chat(
        self,
        *,
        chat_ctx: ChatContext,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        fnc_ctx=None,
        temperature: float | None = None,
        n: int | None = None,
        parallel_tool_calls: bool | None = None,
        tool_choice=None,
    ) -> LLMStream:
        return GroqLLMStream(
            self,
            chat_ctx=chat_ctx,
            fnc_ctx=fnc_ctx,
            conn_options=conn_options,
            temperature=temperature,
            n=n,
            parallel_tool_calls=parallel_tool_calls,
            tool_choice=tool_choice,
        )