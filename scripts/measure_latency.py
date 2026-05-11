import asyncio
import time
import sys
import os
import random
from statistics import median

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

from agent.utils.latency import LatencyTracker
from agent.core.intent import IntentClassifier

TEST_UTTERANCES = [
    "Write a Python function to parse JSON from a URL",
    "Fix the bug in main.py",
    "Run the tests and tell me what fails",
    "Show me the contents of utils.py",
    "What does the requests library do",
    "Stop what you are doing",
    "Add error handling to the parse function",
    "Create a class for database connections",
    "Execute npm install",
    "Read the README file",
]


async def benchmark_intent_classification(n_samples: int = 10) -> None:
    tracker = LatencyTracker()
    classifier = IntentClassifier(tracker)

    print(f"\nBenchmarking intent classification ({n_samples} samples)...")

    for i, utterance in enumerate(TEST_UTTERANCES[:n_samples]):
        start = time.perf_counter()
        result = await classifier.classify(utterance)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(
            f"  [{i+1:2d}]  {elapsed_ms:6.0f}ms  "
            f"{result.intent.value:15s}  {utterance[:50]}"
        )

    print()
    tracker.report()


async def benchmark_simulated_pipeline(n_samples: int = 20) -> None:
    tracker = LatencyTracker()

    print(f"\nSimulating full pipeline timing ({n_samples} samples)...")

    for _ in range(n_samples):
        tracker.record("vad_detection", max(20, random.gauss(60, 15)))
        tracker.record("stt_first_transcript", max(80, random.gauss(180, 40)))
        tracker.record("turn_detection", max(100, random.gauss(220, 40)))
        tracker.record("tts_first_audio", max(150, random.gauss(280, 50)))
        tracker.record("end_to_end", max(500, random.gauss(1200, 120)))

    print()
    tracker.report()


async def main() -> None:
    print("=" * 60)
    print("  Voice-OpenCode Latency Benchmark")
    print("=" * 60)

    await benchmark_simulated_pipeline(20)

    try:
        await benchmark_intent_classification(10)
    except Exception as exc:
        print(f"\nIntent classification benchmark skipped: {exc}")
        print("  Ensure GROQ_API_KEY is set in .env")

    print("\nBenchmark complete.")


if __name__ == "__main__":
    asyncio.run(main())