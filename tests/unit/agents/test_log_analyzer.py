# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LogAnalyzerNode's ReAct loop: a helper tool's failure is structural
(``ToolFailure``), never sniffed from the words of its text."""

from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
import pytest

from artemis.agents.log_analyzer.log_analyzer import LogAnalyzerNode
from artemis.context import ArtemisContext
from artemis.core.tool_failure import ToolFailure
from artemis.tools.mobile.search_logs import SearchLogsArgs

_STATUS_CASES = [
    pytest.param(ToolFailure("Error searching logs: device gone"), "error", id="tool_failure"),
    pytest.param(
        "Error: 12 matching log lines\nE/Foo: Error opening file", "success", id="plain_error_text"
    ),
]


def _text_tool(name: str, result):
    async def _run(keyword: str) -> str:
        return result

    return StructuredTool.from_function(coroutine=_run, name=name, description=name)


@pytest.mark.asyncio
@pytest.mark.parametrize("result, expected_status", _STATUS_CASES)
async def test_agent_loop_tool_message_status_is_structural(result, expected_status):
    ctx = MagicMock(spec=ArtemisContext)
    ctx.data_engine = None
    node = LogAnalyzerNode(ctx=ctx)
    tool = _text_tool("search_logs", result)

    base_llm = MagicMock()
    base_llm.bind_tools.return_value = base_llm
    tool_turn = AIMessage(
        content="", tool_calls=[{"name": "search_logs", "args": {"keyword": "Error"}, "id": "c1"}]
    )
    final_turn = AIMessage(content="Summary.", tool_calls=[])
    messages = [SystemMessage(content="s"), HumanMessage(content="h")]

    with patch(
        "artemis.agents.log_analyzer.log_analyzer.acomplete",
        new=AsyncMock(side_effect=[tool_turn, final_turn]),
    ):
        outcome = await node._run_agent_loop(
            base_llm=base_llm,
            tools_for_binding=[tool],
            tools=[tool],
            current_messages=messages,
            state=MagicMock(),
            max_iterations=5,
            agent_name="Log Analyzer",
        )

    assert outcome == "Summary."
    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "c1"
    assert tool_msgs[0].status == expected_status
    assert tool_msgs[0].content == str(result)


@pytest.mark.asyncio
@pytest.mark.parametrize("raises", [False, True], ids=["typed_failure", "exception"])
async def test_agent_loop_stops_after_three_failed_tool_rounds(raises):
    ctx = MagicMock(spec=ArtemisContext)
    ctx.data_engine = None
    node = LogAnalyzerNode(ctx)
    calls = 0

    async def failing_search(keyword: str) -> str:
        nonlocal calls
        calls += 1
        if raises:
            raise ValueError("Unsupported time format: 17:17:20")
        return ToolFailure("Error searching logs: Unsupported time format: 17:17:20")

    tool = StructuredTool.from_function(
        coroutine=failing_search, name="search_logs", description="Search logs"
    )
    responses = [
        AIMessage(
            content="Need more evidence.",
            tool_calls=[{"name": tool.name, "args": {"keyword": str(i)}, "id": f"c{i}"}],
        )
        for i in range(10)
    ]
    complete = AsyncMock(side_effect=responses)
    with patch("artemis.agents.log_analyzer.log_analyzer.acomplete", complete):
        outcome = await node._run_agent_loop(
            MagicMock(), [tool], [tool], [], MagicMock(), agent_name="Log Analyzer"
        )

    assert isinstance(outcome, ToolFailure)
    assert "3 consecutive" in outcome
    assert "Unsupported time format" in outcome
    assert calls == complete.await_count == 3


@pytest.mark.asyncio
async def test_successful_log_read_resets_failed_round_count():
    ctx = MagicMock(spec=ArtemisContext)
    ctx.data_engine = None
    node = LogAnalyzerNode(ctx)
    results = iter(
        [
            ToolFailure("offline"),
            ToolFailure("offline"),
            "logs",
            ToolFailure("offline"),
            ToolFailure("offline"),
            "logs",
        ]
    )

    async def read(keyword: str) -> str:
        return next(results)

    tool = StructuredTool.from_function(coroutine=read, name="read_logs", description="Read")
    responses = [
        AIMessage(
            content="", tool_calls=[{"name": tool.name, "args": {"keyword": "x"}, "id": f"c{i}"}]
        )
        for i in range(6)
    ] + [AIMessage(content="Recovered diagnosis.")]
    complete = AsyncMock(side_effect=responses)
    with patch("artemis.agents.log_analyzer.log_analyzer.acomplete", complete):
        outcome = await node._run_agent_loop(MagicMock(), [tool], [tool], [], MagicMock())

    assert outcome == "Recovered diagnosis."
    assert complete.await_count == 7


@pytest.mark.asyncio
async def test_cached_search_replays_failed_clock_query_without_retries():
    ctx = MagicMock(spec=ArtemisContext)
    ctx.data_engine = MagicMock(session_start_time=datetime(2026, 10, 20, 17, 15).timestamp())
    node = LogAnalyzerNode(ctx)

    async def unused_search(**kwargs) -> str:
        raise AssertionError("The raw log cache path must handle the search")

    tool = node._wrap_with_caching(
        StructuredTool.from_function(
            coroutine=unused_search,
            name="search_logs",
            description="Search",
            args_schema=SearchLogsArgs,
        )
    )
    turn = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_logs",
                "id": "clock_query",
                "args": {
                    "keyword": "touch",
                    "context_lines": 0,
                    "since_time": "17:15:00.000",
                    "until_time": "17:17:20",
                },
            }
        ],
    )
    complete = AsyncMock(side_effect=[turn, AIMessage(content="Evidence found.")])
    device = MagicMock()
    device.shell.return_value = "10-20 17:15:01.000 touch received\n"
    messages = []
    with (
        patch("artemis.agents.log_analyzer.log_analyzer.acomplete", complete),
        patch("artemis.tools.mobile.log_utils.get_adb_device", return_value=device),
        patch("artemis.tools.mobile.log_utils.datetime", wraps=datetime) as clock,
    ):
        clock.now.return_value = datetime(2026, 10, 21)
        outcome = await node._run_agent_loop(MagicMock(), [tool], [tool], messages, MagicMock())

    assert outcome == "Evidence found."
    assert complete.await_count == 2
    device.shell.assert_called_once_with('logcat -v threadtime -t "10-20 17:15:00.000"')
    tool_message = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_message.status == "success"
    assert "touch received" in tool_message.content
