import asyncio
import time
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps
from statistics import median, quantiles
from typing import Callable

from rich.console import Console
from rich.table import Table

from config.settings import settings


def _latency_logging_enabled() -> bool:
    try:
        return bool(settings.latency_logging)
    except Exception:
        return False


class LatencyTracker:

    TARGETS: dict[str, int] = {
        "vad_detection": 100,
        "stt_first_transcript": 300,
        "turn_detection": 300,
        "llm_first_token": 500,
        "tts_first_audio": 500,
        "end_to_end": 1500,
    }

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def record(self, stage: str, elapsed_ms: float) -> None:
        self._samples[stage].append(elapsed_ms)
        if _latency_logging_enabled():
            from agent.utils.logger import get_logger
            get_logger("latency").info("stage_measured", stage=stage, ms=round(elapsed_ms, 1))

    @contextmanager
    def span(self, stage: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.record(stage, elapsed_ms)

    def report(self) -> None:
        console = Console()
        table = Table(title="Latency Report", show_lines=True)
        table.add_column("Stage", style="bold")
        table.add_column("Target (ms)", justify="right")
        table.add_column("p50 (ms)", justify="right")
        table.add_column("p95 (ms)", justify="right")
        table.add_column("Samples", justify="right")
        table.add_column("Status")

        for stage, samples in sorted(self._samples.items()):
            if not samples:
                continue
            p50 = round(median(samples), 1)
            p95 = (
                round(quantiles(samples, n=20)[18], 1)
                if len(samples) >= 20
                else round(max(samples), 1)
            )
            target = self.TARGETS.get(stage)
            status = ""
            if target:
                status = "pass" if p50 <= target else "over"
            table.add_row(
                stage,
                str(target) if target else "-",
                str(p50),
                str(p95),
                str(len(samples)),
                status,
            )

        console.print(table)


def timed(stage: str, tracker: LatencyTracker) -> Callable:
    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):
            @wraps(fn)
            async def async_wrapper(*args, **kwargs):
                with tracker.span(stage):
                    return await fn(*args, **kwargs)
            return async_wrapper
        else:
            @wraps(fn)
            def sync_wrapper(*args, **kwargs):
                with tracker.span(stage):
                    return fn(*args, **kwargs)
            return sync_wrapper
    return decorator