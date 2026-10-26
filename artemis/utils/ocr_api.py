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

from typing import Any

import httpx
from artemis.config import settings
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)
_HTTP_CLIENT: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _HTTP_CLIENT
    if _HTTP_CLIENT is None or _HTTP_CLIENT.is_closed:
        _HTTP_CLIENT = httpx.AsyncClient(timeout=30.0)
    return _HTTP_CLIENT


def is_ocr_configured() -> bool:
    """OCR shares the same vision model and credentials as the mobile agent."""
    return bool(
        settings.ARTEMIS_OCR_ENABLED
        and settings.OPENAI_BASE_URL
        and settings.OPENAI_API_KEY
        and settings.OPENAI_MODEL
    )


async def perform_ocr(
    screenshot_b64: str,
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Read text and pixel coordinates through the configured vision model."""
    if not is_ocr_configured():
        return []
    from artemis.llm.router import ModelEndpoint, ModelFactory
    from artemis.llm.structured import parse_structured

    import base64
    import io
    from PIL import Image

    image_bytes = base64.b64decode(screenshot_b64)
    image = Image.open(io.BytesIO(image_bytes))
    width, height = image.size
    mime_type = Image.MIME[image.format]
    endpoint = ModelFactory.resolve_endpoint(ModelEndpoint())
    prompt = (
        f"The original screenshot is {width} pixels wide and {height} pixels high. "
        "Read every visible text label in this screenshot. Return only a JSON array "
        'of objects with "text" and "position". "position" must contain four '
        'vertices {"x": integer, "y": integer} in clockwise order, using original '
        "image pixel coordinates. Return [] if there is no visible text."
    )
    active_client = client if client is not None else get_http_client()
    response = await active_client.post(
        endpoint.api_base + "/chat/completions",
        headers={"Authorization": "Bearer " + endpoint.api_key},
        json={
            "model": endpoint.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64," + screenshot_b64},
                        },
                    ],
                }
            ],
        },
        timeout=30.0,
    )
    response.raise_for_status()
    result = parse_structured(response.json()["choices"][0]["message"]["content"])
    if not isinstance(result, list):
        raise ValueError("OCR model did not return a JSON array")
    return [
        item
        for item in result
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and isinstance(item.get("position"), list)
        and len(item["position"]) == 4
        and all(
            isinstance(v, dict) and isinstance(v.get("x"), int) and isinstance(v.get("y"), int)
            for v in item["position"]
        )
    ]
