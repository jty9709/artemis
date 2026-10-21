"""Diagnostic calls must return typed failures and relinquish control on timeout."""

import asyncio
from unittest.mock import AsyncMock, Mock, patch

import pytest

from artemis.core.tool_failure import ToolFailure
from artemis.tools.diagnostic_tool import _run_diagnoser_logic


@pytest.mark.asyncio
async def test_diagnostic_failure_is_not_wrapped_as_a_successful_reply():
    failure = ToolFailure("Stopped after repeated failures")
    with patch("artemis.tools.diagnostic_tool.Diagnoser") as diagnoser:
        diagnoser.return_value.run = AsyncMock(return_value=failure)
        outcome = await _run_diagnoser_logic(Mock(), Mock(), "diagnose")
    assert outcome is failure


@pytest.mark.asyncio
async def test_diagnostic_exception_returns_typed_failure():
    with patch("artemis.tools.diagnostic_tool.Diagnoser") as diagnoser:
        diagnoser.return_value.run = AsyncMock(side_effect=RuntimeError("device offline"))
        outcome = await _run_diagnoser_logic(Mock(), Mock(), "diagnose")
    assert isinstance(outcome, ToolFailure)
    assert "device offline" in outcome


@pytest.mark.asyncio
async def test_nested_timeout_is_not_misreported_as_the_overall_deadline():
    with patch("artemis.tools.diagnostic_tool.Diagnoser") as diagnoser:
        diagnoser.return_value.run = AsyncMock(side_effect=TimeoutError("log fetch timed out"))
        outcome = await _run_diagnoser_logic(Mock(), Mock(), "diagnose")
    assert isinstance(outcome, ToolFailure)
    assert "log fetch timed out" in outcome
    assert "120 seconds" not in outcome


@pytest.mark.asyncio
async def test_diagnostic_timeout_cancels_nested_work_and_returns_control():
    cancelled = asyncio.Event()

    async def stuck_run(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    with (
        patch("artemis.tools.diagnostic_tool.Diagnoser") as diagnoser,
        patch("artemis.tools.diagnostic_tool._DIAGNOSIS_TIMEOUT_SECONDS", 0.01),
    ):
        diagnoser.return_value.run = AsyncMock(side_effect=stuck_run)
        outcome = await _run_diagnoser_logic(Mock(), Mock(), "diagnose")

    assert isinstance(outcome, ToolFailure)
    assert "timed out" in outcome
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_external_cancellation_is_not_swallowed():
    with patch("artemis.tools.diagnostic_tool.Diagnoser") as diagnoser:
        diagnoser.return_value.run = AsyncMock(side_effect=asyncio.CancelledError)
        with pytest.raises(asyncio.CancelledError):
            await _run_diagnoser_logic(Mock(), Mock(), "diagnose")
