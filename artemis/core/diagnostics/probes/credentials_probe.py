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

"""LLM & Multimodal Vision Credentials Readiness Probe."""

from artemis.config import settings
from artemis.core.diagnostics.probes.base import BaseProbe
from artemis.core.diagnostics.schema import (
    ProbeCategory,
    ProbeResult,
    ProbeStatus,
)


class LLMCredentialsProbe(BaseProbe):
    """Probe for the single configured OpenAI-compatible service."""

    @property
    def probe_id(self) -> str:
        return "gemini_api_key"

    @property
    def category(self) -> ProbeCategory:
        return ProbeCategory.CREDENTIALS

    @property
    def is_blocker(self) -> bool:
        return True

    def _mask_key(self, key_str: str) -> str:
        """Helper to safely mask an API credential for display."""
        if len(key_str) > 10:
            return f"****{key_str[-4:]}"
        return "***"

    async def probe(self) -> ProbeResult:
        key = settings.get_api_key("openai")
        missing = [
            name
            for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL")
            if not getattr(settings, name)
        ]
        provider = {
            "provider": "openai",
            "label": "OpenAI-compatible service",
            "masked": self._mask_key(key.get_secret_value()) if key else "",
        }
        return ProbeResult(
            id=self.probe_id,
            category=self.category,
            title="OpenAI-compatible model",
            status=ProbeStatus.FAIL if missing else ProbeStatus.PASS,
            is_blocker=self.is_blocker,
            summary="Missing: " + ", ".join(missing)
            if missing
            else f"Active ({settings.OPENAI_MODEL})",
            description="Configure OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL in .env.",
            metadata={
                "configured_count": 0 if missing else 1,
                "providers": [] if missing else [provider],
                "base_url": settings.OPENAI_BASE_URL,
                "model": settings.OPENAI_MODEL,
            },
            actions=[],
        )


class VisionOCRProbe(BaseProbe):
    """Probe for OCR through the configured vision model."""

    @property
    def probe_id(self) -> str:
        return "vision_ocr_key"

    @property
    def category(self) -> ProbeCategory:
        return ProbeCategory.CREDENTIALS

    @property
    def is_blocker(self) -> bool:
        return False

    async def probe(self) -> ProbeResult:
        from artemis.utils.ocr_api import is_ocr_configured

        return ProbeResult(
            id=self.probe_id,
            category=self.category,
            title="OCR model",
            status=ProbeStatus.PASS
            if not settings.ARTEMIS_OCR_ENABLED or is_ocr_configured()
            else ProbeStatus.WARN,
            is_blocker=False,
            summary="Uses the configured vision model"
            if settings.ARTEMIS_OCR_ENABLED
            else "Disabled (optional)",
            description="OCR shares OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL; no separate key is needed.",
            metadata={"configured": is_ocr_configured(), "enabled": settings.ARTEMIS_OCR_ENABLED},
            actions=[],
        )
