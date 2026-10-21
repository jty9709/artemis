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

from datetime import datetime
from unittest.mock import Mock, patch
from artemis.context import ArtemisContext
from artemis.tools.mobile.log_utils import fetch_and_filter_logs, resolve_time
import pytest


@pytest.fixture
def mock_context():
    ctx = Mock(spec=ArtemisContext)
    ctx.data_engine = Mock()
    ctx.data_engine.session_start_time = 1700000000.0  # Example timestamp
    return ctx


@pytest.fixture
def mock_device():
    return Mock()


@patch("artemis.tools.mobile.log_utils.get_adb_device")
def test_fetch_and_filter_logs_basic(mock_get_adb_device, mock_context, mock_device):
    mock_get_adb_device.return_value = mock_device
    mock_device.shell.return_value = "line 1\nline 2\n"

    result = fetch_and_filter_logs(mock_context, lines=10)

    mock_device.shell.assert_called_once_with("logcat -v threadtime -t 10")
    assert result == "line 1\nline 2\n"


@patch("artemis.tools.mobile.log_utils.get_adb_device")
@patch("artemis.tools.mobile.log_utils.resolve_time")
def test_fetch_and_filter_logs_since_time_success(
    mock_resolve_time, mock_get_adb_device, mock_context, mock_device
):
    mock_get_adb_device.return_value = mock_device
    since_dt = datetime(datetime.now().year, 5, 3, 11, 11, 11, 111000)
    mock_resolve_time.side_effect = [
        ("05-03 11:11:11.111", since_dt),  # for since_time
        (None, None),  # for until_time
    ]

    mock_device.shell.return_value = "05-03 11:11:12.000 line 1\n05-03 11:11:13.000 line 2\n"

    result = fetch_and_filter_logs(mock_context, since_time="05-03 11:11:11.111")

    mock_device.shell.assert_called_once_with('logcat -v threadtime -t "05-03 11:11:11.111"')
    assert "line 1" in result
    assert "line 2" in result


@patch("artemis.tools.mobile.log_utils.get_adb_device")
@patch("artemis.tools.mobile.log_utils.resolve_time")
def test_fetch_and_filter_logs_since_time_fallback(
    mock_resolve_time, mock_get_adb_device, mock_context, mock_device
):
    mock_get_adb_device.return_value = mock_device
    current_year = datetime.now().year
    since_dt = datetime(current_year, 5, 4, 12, 0, 0)
    mock_resolve_time.side_effect = [
        ("05-04 12:00:00.000", since_dt),  # for since_time
        (None, None),  # for until_time
    ]

    # Direct fetch fails
    mock_device.shell.side_effect = [
        Exception("logcat failed"),  # first call to direct fetch
        "05-04 11:59:00.000 line old\n05-04 12:00:01.000 line new\n",  # second call (fallback)
    ]

    result = fetch_and_filter_logs(mock_context, lines=100, since_time="05-04 12:00:00.000")

    assert mock_device.shell.call_count == 2
    mock_device.shell.assert_any_call('logcat -v threadtime -t "05-04 12:00:00.000"')
    mock_device.shell.assert_any_call("logcat -v threadtime -t 100")

    # Should filter out "line old" because it's before since_dt
    assert "line old" not in result
    assert "line new" in result


@patch("artemis.tools.mobile.log_utils.get_adb_device")
@patch("artemis.tools.mobile.log_utils.resolve_time")
def test_fetch_and_filter_logs_until_time(
    mock_resolve_time, mock_get_adb_device, mock_context, mock_device
):
    mock_get_adb_device.return_value = mock_device
    current_year = datetime.now().year
    until_dt = datetime(current_year, 5, 4, 12, 0, 0)
    mock_resolve_time.side_effect = [
        (None, None),  # for since_time
        ("05-04 12:00:00.000", until_dt),  # for until_time
    ]

    mock_device.shell.return_value = "05-04 11:59:00.000 line old\n05-04 12:00:01.000 line new\n"

    result = fetch_and_filter_logs(mock_context, lines=100, until_time="05-04 12:00:00.000")

    mock_device.shell.assert_called_once_with("logcat -v threadtime -t 100")
    assert "line old" in result
    assert "line new" not in result


@pytest.mark.parametrize(
    "clock, session_date, expected",
    [
        ("17:15:00.000", datetime(2026, 10, 20, 16), datetime(2026, 10, 20, 17, 15)),
        ("17:14:30.000", datetime(2026, 10, 20, 16), datetime(2026, 10, 20, 17, 14, 30)),
        ("17:17:20", datetime(2026, 10, 20, 16), datetime(2026, 10, 20, 17, 17, 20)),
        (" 17:17:20.123 ", datetime(2026, 10, 20, 16), datetime(2026, 10, 20, 17, 17, 20, 123000)),
        ("00:01:00", datetime(2026, 12, 31, 23, 59), datetime(2027, 1, 1, 0, 1)),
        ("23:59:00", datetime(2027, 1, 1, 0, 1), datetime(2026, 12, 31, 23, 59)),
    ],
)
def test_resolve_clock_time_uses_session_date(clock, session_date, expected):
    formatted, resolved = resolve_time(clock, session_date.timestamp())

    assert resolved == expected
    assert formatted == expected.strftime("%m-%d %H:%M:%S.%f")[:-3]


@pytest.mark.parametrize("seconds", ["15.5", 15.5])
def test_resolve_time_keeps_relative_seconds(seconds):
    start = datetime(2026, 10, 20, 16).timestamp()
    assert resolve_time(seconds, start) == (
        "10-20 16:00:15.500",
        datetime(2026, 10, 20, 16, 0, 15, 500000),
    )


def test_resolve_time_keeps_absolute_logcat_timestamps():
    with patch("artemis.tools.mobile.log_utils.datetime", wraps=datetime) as clock:
        clock.now.return_value = datetime(2026, 10, 21)
        assert resolve_time("10-20 17:15:00.123", None) == (
            "10-20 17:15:00.123",
            datetime(2026, 10, 20, 17, 15, 0, 123000),
        )


def test_resolve_clock_without_session_uses_nearest_current_date():
    with patch("artemis.tools.mobile.log_utils.datetime", wraps=datetime) as clock:
        clock.now.return_value = datetime(2026, 10, 21, 0, 1)
        assert resolve_time("23:59:00", None) == (
            "10-20 23:59:00.000",
            datetime(2026, 10, 20, 23, 59),
        )


@pytest.mark.parametrize("value", ["25:00:00", "17:15", "yesterday"])
def test_resolve_time_rejects_invalid_input_with_format_guidance(value):
    with pytest.raises(ValueError, match="Use .*HH:MM:SS"):
        resolve_time(value, datetime(2026, 10, 20, 16).timestamp())


@patch("artemis.tools.mobile.log_utils.get_adb_device")
def test_fetch_and_filter_logs_accepts_original_failed_clock_arguments(
    mock_get_adb_device, mock_context, mock_device
):
    mock_context.data_engine.session_start_time = datetime(2026, 10, 20, 16).timestamp()
    mock_get_adb_device.return_value = mock_device
    mock_device.shell.return_value = (
        "10-20 17:14:59.999 before\n10-20 17:15:00.001 wanted\n10-20 17:17:20.001 after\n"
    )
    with patch("artemis.tools.mobile.log_utils.datetime", wraps=datetime) as clock:
        clock.now.return_value = datetime(2026, 10, 21)
        result = fetch_and_filter_logs(
            mock_context, since_time="17:15:00.000", until_time="17:17:20"
        )

    mock_device.shell.assert_called_once_with('logcat -v threadtime -t "10-20 17:15:00.000"')
    assert result == "10-20 17:15:00.001 wanted"
