from __future__ import annotations

from dataclasses import dataclass

from transformers import pipeline

from app.core.config import settings


@dataclass(frozen=True)
class QaResult:
    answer: str
    score: float | None


class QAService:
    """
    Singleton QA pipeline loaded once at startup.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._pipe = None

    def load(self) -> None:
        if self._pipe is None:
            self._pipe = pipeline(
                "question-answering",
                model=self.model_name,
                tokenizer=self.model_name,
            )

    def answer(self, question: str, context: str) -> QaResult:
        if self._pipe is None:
            raise RuntimeError("QA model not loaded. Call load() first.")

        out = self._pipe(question=question, context=context)
        ans = (out.get("answer") or "").strip()
        score = out.get("score", None)

        try:
            score_f = float(score) if score is not None else None
        except Exception:
            score_f = None

        return QaResult(answer=ans, score=score_f)


def default_qa_service() -> QAService:
    return QAService(settings.QA_MODEL_NAME)
