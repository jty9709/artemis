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

"""Repository-wide pytest fixtures and classification helpers.

The default test paths contain only deterministic tests.  Tests under the
integration and end-to-end trees remain directly runnable, and receive stable
markers here so callers can select them without relying on filename patterns.
"""

from pathlib import Path

import pytest

from artemis.drivers.mock.mock_driver import MockDeviceDriver


@pytest.fixture
def mock_driver():
    """Provide an isolated mock mobile driver."""
    return MockDeviceDriver(device_id="fixture-mock-device", width=1080, height=2400)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Attach test-layer markers according to the owning test directory."""
    for item in items:
        parts = Path(str(item.path)).parts
        if "integration" in parts:
            item.add_marker(pytest.mark.integration)
        if "e2e" in parts:
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(autouse=True)
def isolate_unit_model_credentials(request, monkeypatch):
    """Never send local credentials to a service or include them in unit-test failures."""
    if "unit" not in Path(str(request.fspath)).parts:
        return
    from pydantic import SecretStr
    from artemis.config import settings

    for name, value in {
        "OPENAI_BASE_URL": "http://127.0.0.1:9/v1",
        "OPENAI_API_KEY": "unit-test-key",
        "OPENAI_MODEL": "unit-test-vision",
    }.items():
        monkeypatch.setenv(name, value)
        monkeypatch.setattr(settings, name, SecretStr(value) if name.endswith("KEY") else value)
