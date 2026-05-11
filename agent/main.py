import asyncio
import signal

from livekit.agents import WorkerOptions, cli

from agent.core.outer_loop import entrypoint
from agent.utils.logger import configure_logging, get_logger
from agent.utils.latency import LatencyTracker
from config.settings import settings

logger = get_logger(__name__)
_tracker = LatencyTracker()


def main() -> None:
    configure_logging()

    logger.info(
        "voice_opencode_starting",
        livekit_url=settings.livekit_url,
        outer_loop_model=settings.outer_loop_model,
        subagent_model=settings.subagent_model,
        workspace=settings.opencode_workspace,
    )

    print(
        "\n"
        "  Voice-OpenCode Agent\n"
        "  Speak to code. Talk to your terminal.\n"
        f"  LiveKit    -> {settings.livekit_url}\n"
        f"  Workspace  -> {settings.opencode_workspace}\n"
        f"  Outer loop -> {settings.outer_loop_model}\n"
        f"  Subagents  -> {settings.subagent_model}\n"
    )

    def _on_shutdown(signum, frame):
        print("\nLatency report:")
        _tracker.report()
        exit(0)

    signal.signal(signal.SIGINT, _on_shutdown)
    signal.signal(signal.SIGTERM, _on_shutdown)

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=settings.livekit_api_key,
            api_secret=settings.livekit_api_secret,
            ws_url=settings.livekit_url,
        )
    )


if __name__ == "__main__":
    main()