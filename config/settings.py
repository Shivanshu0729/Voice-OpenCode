from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    livekit_url: str = Field(...)
    livekit_api_key: str = Field(...)
    livekit_api_secret: str = Field(...)

    deepgram_api_key: str = Field(...)
    deepgram_model: str = Field("nova-3")
    deepgram_diarize: bool = Field(True)

    groq_api_key: str = Field(...)
    outer_loop_model: str = Field("llama-3.1-8b-instant")
    subagent_model: str = Field("llama-3.1-8b-instant")

    cartesia_api_key: str = Field(...)
    cartesia_voice_id: str = Field("a0e99841-438c-4a64-b679-ae501e7d6091")

    opencode_workspace: str = Field(".")
    opencode_binary: str = Field("opencode")
    subagent_timeout: int = Field(120, ge=10, le=600)

    log_level: str = Field("INFO")
    latency_logging: bool = Field(True)

    @field_validator("log_level")
    @classmethod
    def normalise_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return upper

    @field_validator("livekit_url")
    @classmethod
    def validate_livekit_url(cls, v: str) -> str:
        if not v.startswith(("ws://", "wss://")):
            raise ValueError("LIVEKIT_URL must start with ws:// or wss://")
        return v




@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


class _SettingsProxy:
    def __getattr__(self, item: str):
        return getattr(get_settings(), item)

    def __repr__(self) -> str:
        return "SettingsProxy()"


settings = _SettingsProxy()