import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.subagents.pool import SubagentPool, TaskType, SubagentResult


class TestSubagentPool:
    @pytest.mark.asyncio
    async def test_dispatch_returns_task_id_immediately(self):
        pool = SubagentPool()
        with patch.object(pool, "_run_task", new=AsyncMock()):
            task_id = pool.dispatch(TaskType.CODING, "write a function")
        assert isinstance(task_id, str)
        assert len(task_id) == 8

    @pytest.mark.asyncio
    async def test_cancel_specific_task_removes_it(self):
        pool = SubagentPool()

        async def _slow_task(task_id, task_type, prompt, file_hint):
            await asyncio.sleep(60)

        with patch.object(pool, "_run_task", side_effect=_slow_task):
            task_id = pool.dispatch(TaskType.CODING, "slow task")

        await asyncio.sleep(0.05)
        cancelled = pool.cancel(task_id)
        assert cancelled == 1
        assert pool.active_count == 0

    @pytest.mark.asyncio
    async def test_cancel_all_tasks(self):
        pool = SubagentPool()

        async def _slow_task(task_id, task_type, prompt, file_hint):
            await asyncio.sleep(60)

        with patch.object(pool, "_run_task", side_effect=_slow_task):
            pool.dispatch(TaskType.CODING, "task 1")
            pool.dispatch(TaskType.RUN_COMMAND, "task 2")

        await asyncio.sleep(0.05)
        cancelled = pool.cancel()
        assert cancelled == 2
        assert pool.active_count == 0

    @pytest.mark.asyncio
    async def test_drain_results_returns_empty_list_when_no_results(self):
        pool = SubagentPool()
        results = await pool.drain_results()
        assert results == []

    @pytest.mark.asyncio
    async def test_completed_result_appears_in_drain(self):
        pool = SubagentPool()

        await pool._results.put(
            SubagentResult(
                task_id="abc12345",
                task_type=TaskType.CODING,
                success=True,
                summary="Done. Added parse_json() to utils.py",
            )
        )

        results = await pool.drain_results()
        assert len(results) == 1
        assert results[0].success is True
        assert "parse_json" in results[0].summary

    @pytest.mark.asyncio
    async def test_cancelled_result_has_cancelled_flag(self):
        pool = SubagentPool()

        await pool._results.put(
            SubagentResult(
                task_id="xyz99999",
                task_type=TaskType.BUG_FIX,
                success=False,
                summary="Task cancelled.",
                cancelled=True,
            )
        )

        results = await pool.drain_results()
        assert results[0].cancelled is True


class TestCommandExecutor:
    @pytest.mark.asyncio
    async def test_echo_command_succeeds(self):
        from agent.subagents.executer import CommandExecutor
        executor = CommandExecutor()
        result = await executor.run("echo voice_opencode_test")
        assert result.success
        assert "voice_opencode_test" in result.stdout

    @pytest.mark.asyncio
    async def test_nonexistent_command_returns_failure(self):
        from agent.subagents.executer import CommandExecutor
        executor = CommandExecutor()
        result = await executor.run("this_binary_does_not_exist_xyz123")
        assert not result.success

    @pytest.mark.asyncio
    async def test_progress_callback_is_called(self):
        from agent.subagents.executer import CommandExecutor
        received_lines = []

        async def capture(line: str):
            received_lines.append(line)

        executor = CommandExecutor(on_progress=capture)
        await executor.run("echo hello_progress")
        assert any("hello_progress" in line for line in received_lines)