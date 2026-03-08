from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.core.errors import ExternalDependencyMissing
from app.models.ner import Entity
from app.services.retrieval.retriever import RetrievedChunk


@dataclass(frozen=True)
class _RawEnt:
    text: str
    label: str
    start: int
    end: int


class NerService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._nlp = None

    def load(self) -> None:
        if self._nlp is None:
            try:
                import spacy
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise ExternalDependencyMissing("spacy") from exc
            self._nlp = spacy.load(self.model_name)

    def _extract_from_text(self, text: str) -> list[_RawEnt]:
        if self._nlp is None:
            raise RuntimeError("NER model not loaded. Call load() first.")
        doc = self._nlp(text)
        return [
            _RawEnt(
                text=ent.text,
                label=ent.label_,
                start=int(ent.start_char),
                end=int(ent.end_char),
            )
            for ent in doc.ents
        ]

    @staticmethod
    def _dedupe_and_cap(entities: list[Entity]) -> list[Entity]:
        seen: set[tuple[str, str, str, str | None]] = set()
        out: list[Entity] = []
        for entity in entities:
            key = (
                entity.text.strip().lower(),
                entity.label,
                entity.source,
                entity.chunk_id,
            )
            if key in seen:
                continue
            seen.add(key)
            out.append(entity)
            if len(out) >= settings.MAX_ENTITIES:
                break
        return out

    def extract_entities(
        self, answer: str, sources: list[RetrievedChunk]
    ) -> list[Entity]:
        all_entities: list[Entity] = []

        answer_text = (answer or "").strip()
        if answer_text:
            for ent in self._extract_from_text(answer_text):
                all_entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label,
                        start=ent.start,
                        end=ent.end,
                        source="answer",
                        doc_id=None,
                        page=None,
                        chunk_id=None,
                    )
                )

        for source in sources:
            chunk_text = (source.text or source.text_snippet or "").strip()
            if not chunk_text:
                continue
            for ent in self._extract_from_text(chunk_text):
                all_entities.append(
                    Entity(
                        text=ent.text,
                        label=ent.label,
                        start=ent.start,
                        end=ent.end,
                        source="chunk",
                        doc_id=source.doc_id,
                        page=source.page,
                        chunk_id=source.chunk_id,
                    )
                )

        return self._dedupe_and_cap(all_entities)


def default_ner_service() -> NerService:
    return NerService(settings.NER_MODEL_NAME)
