from __future__ import annotations

import logging
import re
from typing import Any

import requests

from app.core.config import settings
from app.core.errors import ServiceUnavailable
from app.services.interfaces import QaResult

CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
DEFAULT_SYSTEM_PROMPT = (
    "You answer questions about uploaded business documents using only the evidence "
    "provided by the application. Keep answers concise, natural, and directly useful. "
    "Do not dump raw OCR text. If the question asks for one field, answer that field "
    "first in one sentence. If the question asks for a summary, produce a brief plain-"
    "English summary in at most three sentences. If the evidence is limited, answer with "
    "what is supported and avoid speculation. Return plain text only."
)
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

logger = logging.getLogger(__name__)


class QAService:
    """Optional OpenAI-backed answer synthesis service with safe local fallback."""

    def __init__(
        self,
        model_name: str,
        *,
        use_openai: bool,
        api_key: str | None,
        base_url: str | None,
        organization: str | None,
        timeout_seconds: int,
        max_output_tokens: int,
    ):
        self.model_name = model_name
        self.use_openai = use_openai
        self.api_key = (api_key or "").strip() or None
        self.base_url = (base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self.organization = organization
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self._session: requests.Session | None = None
        self.backend = "heuristic"
        self.status_detail = "heuristic fallback only"

    @property
    def openai_enabled(self) -> bool:
        return self.use_openai and bool(self.api_key)

    def load(self) -> None:
        if self._session is not None:
            return

        if not self.use_openai:
            self.backend = "heuristic"
            self.status_detail = "OpenAI synthesis disabled by configuration"
            logger.info(
                "QA service running in heuristic-only mode (QA_USE_OPENAI=false)."
            )
            return

        if not self.api_key:
            self.backend = "heuristic"
            self.status_detail = (
                "OPENAI_API_KEY not configured; using heuristic fallback"
            )
            logger.warning(
                "OPENAI_API_KEY is not configured. Falling back to heuristic answer synthesis."
            )
            return

        session = requests.Session()
        session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        if self.organization:
            session.headers["OpenAI-Organization"] = self.organization

        self._session = session
        self.backend = "openai"
        self.status_detail = "OpenAI Responses API initialized"

    def answer(self, question: str, context: str) -> QaResult:
        if self._session is None:
            return QaResult(answer="", score=None)

        prompt = (
            f"Question:\n{question.strip()}\n\n"
            "Application context:\n"
            f"{context.strip()}"
        )
        payload = {
            "model": self.model_name,
            "instructions": DEFAULT_SYSTEM_PROMPT,
            "input": prompt,
            "max_output_tokens": int(self.max_output_tokens),
        }

        try:
            response = self._session.post(
                f"{self.base_url}/responses",
                json=payload,
                timeout=float(self.timeout_seconds),
            )
        except requests.RequestException as exc:
            raise ServiceUnavailable(
                f"OpenAI answer synthesis request failed: {type(exc).__name__}."
            ) from exc

        if response.status_code >= 400:
            error_message = _extract_error_message(response)
            raise ServiceUnavailable(
                f"OpenAI answer synthesis request failed: {error_message}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ServiceUnavailable(
                "OpenAI answer synthesis response was not valid JSON."
            ) from exc

        answer = _extract_response_text(payload)
        return QaResult(answer=answer, score=None)


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"

    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if payload.get("message"):
            return str(payload["message"])
    return f"HTTP {response.status_code}"


def _extract_response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return _clean_output_text(output_text)

    output = payload.get("output")
    if isinstance(output, list):
        extracted = _extract_text_from_output_items(output)
        if extracted:
            return _clean_output_text(extracted)

    return ""


def _extract_text_from_output_items(items: list[Any]) -> str:
    collected: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                collected.append(text.strip())
            elif isinstance(part.get("output_text"), str):
                collected.append(str(part["output_text"]).strip())
    return "\n".join(collected).strip()


def _clean_output_text(text: str) -> str:
    cleaned = CODE_FENCE_RE.sub("", (text or "").strip()).strip()
    return cleaned


def default_qa_service() -> QAService:
    return QAService(
        settings.QA_MODEL_NAME,
        use_openai=settings.QA_USE_OPENAI,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        organization=settings.OPENAI_ORGANIZATION,
        timeout_seconds=settings.OPENAI_TIMEOUT_SECONDS,
        max_output_tokens=settings.OPENAI_MAX_OUTPUT_TOKENS,
    )
