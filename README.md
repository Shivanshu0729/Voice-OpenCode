# Voice OpenCode

Voice OpenCode is a LiveKit-based voice coding agent. It listens to speech, classifies the intent, routes work to subagents, and speaks back using TTS.

## What You Need

- Python 3.11 or newer
- Windows, macOS, or Linux
- A LiveKit room, API key, and API secret
- Deepgram, Groq, and Cartesia API keys
- The OpenCode CLI available on `PATH`

## Quick Start

The goal is to get from a fresh machine to a working agent in under 10 minutes.

1. Clone the repository and open it in VS Code.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux, use:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```powershell
pip install -r requirements.txt -r requirements-dev.txt
```

4. Create a `.env` file in the project root with these variables:

```dotenv
LIVEKIT_URL=wss://your-livekit-host
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret

DEEPGRAM_API_KEY=your_deepgram_key
DEEPGRAM_MODEL=nova-3
DEEPGRAM_DIARIZE=true

GROQ_API_KEY=your_groq_key
OUTER_LOOP_MODEL=llama-3.1-8b-instant
SUBAGENT_MODEL=llama-3.1-8b-instant

CARTESIA_API_KEY=your_cartesia_key
CARTESIA_VOICE_ID=a0e99841-438c-4a64-b679-ae501e7d6091

OPENCODE_WORKSPACE=.
OPENCODE_BINARY=opencode
SUBAGENT_TIMEOUT=120

LOG_LEVEL=INFO
LATENCY_LOGGING=true
```

5. Make sure the OpenCode CLI works from your terminal:

```powershell
opencode --help
```

6. Generate a LiveKit token for the demo room:

```powershell
python generate_token.py
```

7. Start the agent:

```powershell
python -m agent.main connect --room voice-opencode-test --no-watch
```

8. Open the LiveKit Agents Playground, join the same room, and speak to the agent.

## How It Works

- VAD detects when you start and stop speaking.
- Deepgram transcribes your speech.
- Groq classifies the intent.
- The outer loop responds immediately and dispatches subagents when needed.
- Cartesia is used for primary TTS, with a local Windows fallback when Cartesia is unavailable.

## Run Tests

```powershell
pytest
```

## Useful Commands

```powershell
python generate_token.py
python -m agent.main connect --room voice-opencode-test --no-watch
pytest tests/test_outer_loop.py -vv
```

## Troubleshooting

- If the agent connects but stays silent, make sure the playground room name matches the room in `.env` and the startup command.
- If Cartesia returns `402 Payment Required`, the agent falls back to local Windows TTS on Windows.
- If the room connection keeps timing out, verify the LiveKit URL, API key, and API secret in `.env`.
- If `opencode` is not found, install the OpenCode CLI or add it to your `PATH`.

## Project Structure

```text
VOICE-OPENCODE/
│
├── agent/
│   ├── core/
│   │   ├── intent.py
│   │   ├── llm_adapter.py
│   │   ├── outer_loop.py
│   │   ├── stt.py
│   │   ├── tts.py
│   │   ├── turn_detector.py
│   │   └── vad.py
│   │
│   ├── subagents/
│   │   ├── executer.py
│   │   ├── filesystem.py
│   │   ├── opencode.py
│   │   └── pool.py
│   │
│   ├── utils/
│   │   ├── latency.py
│   │   └── logger.py
│   │
│   └── main.py
│
├── config/
│   └── settings.py
│
├── scripts/
│   └── measure_latency.py
│
├── tests/
│   ├── __init__.py
│   ├── test_opencode.py
│   ├── test_outer_loop.py
│   ├── test_subagents.py
│   └── test_tts.py
│
├── .env
├── .env.example
├── .gitignore
├── generate_token.py
├── measure_latency.py
├── README.md
├── requirements.txt
└── requirements-dev.txt
```
