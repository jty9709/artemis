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

"""Validation utility to verify LLM and Vision API keys against live provider endpoints."""

import httpx

from artemis.config import settings
from artemis.config.settings import is_placeholder_key

from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)


def _extract_error_message(resp: httpx.Response) -> str:
    """Extract a human-friendly error message from a provider HTTP response."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            # Google style: {"error": {"message": "..."}}
            err_obj = data.get("error")
            if isinstance(err_obj, dict):
                msg = err_obj.get("message")
                if msg:
                    return str(msg)
            elif isinstance(err_obj, str):
                return err_obj

            # OpenAI / Anthropic / OpenRouter style
            if "message" in data:
                return str(data["message"])
            if "detail" in data:
                return str(data["detail"])
    except ValueError:
        # Response body is not valid JSON; fall through to the text fallback.
        pass

    # Fallback to response text or HTTP status
    text = resp.text.strip()
    if text and len(text) < 200:
        return text
    return f"HTTP {resp.status_code}: {resp.reason_phrase}"


async def validate_api_key(
    provider: str,
    api_key: str,
    base_url: str | None = None,
    timeout: float = 12.0,
) -> tuple[bool, str]:
    """Tests if the provided API key is valid and usable with the corresponding provider.

    Args:
        provider: Provider identifier (e.g. 'google', 'gemini', 'ocr', 'openai', 'anthropic', etc.)
        api_key: Secret API key string.
        base_url: Optional custom endpoint URL.
        timeout: Request timeout in seconds.

    Returns:
        tuple[bool, str]: (is_valid, descriptive_message)
    """
    if provider.strip().lower() != "openai":
        return False, "Only the configured OpenAI-compatible service is enabled."
    clean_key = api_key.strip()
    if not clean_key:
        return False, "API key cannot be empty."
    if is_placeholder_key(clean_key):
        return False, "Please configure a real OPENAI_API_KEY."
    endpoint = base_url or settings.OPENAI_BASE_URL
    if not endpoint:
        return False, "Configure OPENAI_BASE_URL in .env."
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                endpoint.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {clean_key}"},
            )
        if resp.status_code == 200:
            models = [item.get("id") for item in resp.json().get("data", [])]
            if settings.OPENAI_MODEL and settings.OPENAI_MODEL not in models:
                return False, "OPENAI_MODEL is not listed by the configured service."
            return True, "Configured OpenAI-compatible service and model verified."
        return (
            False,
            f"API verification failed ({resp.status_code}): {_extract_error_message(resp)}",
        )

    except httpx.TimeoutException:
        logger.warning(f"Timeout verifying API key for {provider}")
        return (
            False,
            f"Connection to {provider.capitalize()} API timed out. Please check network/proxy settings.",
        )
    except httpx.RequestError as exc:
        logger.warning(f"Network error verifying API key for {provider}: {exc}")
        return False, f"Network error connecting to {provider.capitalize()} API: {exc}"
    except Exception as exc:
        logger.error(f"Unexpected error validating API key for {provider}: {exc}")
        return False, f"Validation error: {exc}"
