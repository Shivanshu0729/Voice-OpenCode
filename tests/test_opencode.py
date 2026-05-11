import asyncio
import json
import pytest
from unittest.mock import patch

from agent.subagents.opencode import OpenCodeSession, EventType, OpenCodeEvent


class TestOpenCodeEventParsing:
    def test_parses_progress_event(self):
        session = OpenCodeSession()
        line = json.dumps({"type": "progress", "message": "Writing function..."})
        event = session._parse_line(line)
        assert event.type == EventType.PROGRESS
        assert event.message == "Writing function..."

    def test_parses_file_changed_event_with_line_number(self):
        session = OpenCodeSession()
        line = json.dumps({"type": "file_changed", "path": "utils.py", "line": 42})
        event = session._parse_line(line)
        assert event.type == EventType.FILE_CHANGED
        assert event.path == "utils.py"
        assert event.line == 42

    def test_parses_result_event_with_success_flag(self):
        session = OpenCodeSession()
        line = json.dumps({
            "type": "result",
            "summary": "Done. Added parse_json().",
            "success": True,
        })
        event = session._parse_line(line)
        assert event.type == EventType.RESULT
        assert event.success is True

    def test_parses_error_event(self):
        session = OpenCodeSession()
        line = json.dumps({"type": "error", "message": "File not found: main.py"})
        event = session._parse_line(line)
        assert event.type == EventType.ERROR
        assert "not found" in event.message

    def test_non_json_line_falls_back_to_raw(self):
        session = OpenCodeSession()
        event = session._parse_line("Starting OpenCode session...")
        assert event.type == EventType.RAW
        assert "Starting" in event.message

    def test_empty_type_defaults_to_raw(self):
        session = OpenCodeSession()
        event = session._parse_line(json.dumps({"type": "unknown_future_type", "message": "x"}))
        assert event.type == EventType.RAW


class TestOpenCodeAvailability:
    @pytest.mark.asyncio
    async def test_yields_error_event_when_binary_missing(self):
        session = OpenCodeSession()
        with patch.object(OpenCodeSession, "is_available", return_value=False):
            events = []
            async for event in session.run("write something"):
                events.append(event)

        assert len(events) == 1
        assert events[0].type == EventType.ERROR
        assert "not found" in events[0].message.lower()

    def test_is_available_returns_bool(self):
        result = OpenCodeSession.is_available()
        assert isinstance(result, bool)