from __future__ import annotations

from dataclasses import dataclass

from app.core.config import settings
from app.models.ner import Entity
from app.services.retrieval.retriever import RetrievedChunk


@dataclass(frozen=True)
class _RawEnt:
    text: str
    label: str
    start: int
    end: int


class NerService:
    """
    Singleton spaCy pipeline.
    Fail-soft: if model isn't available, keep service disabled (None in app.state)
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._nlp = None

    def load(self) -> None:
        if self._nlp is None:
            import spacy

            self._nlp = spacy.load(self.model_name)

    def _extract_from_text(self, text: str) -> list[_RawEnt]:
        if self._nlp is None:
            raise RuntimeError("NER model not loaded. Call load() first.")

        doc = self._nlp(text)
        out: list[_RawEnt] = []

        for ent in doc.ents:
            out.append(
                _RawEnt(
                    text=ent.text,
                    label=ent.label_,
                    start=int(ent.start_char),
                    end=int(ent.end_char),
                )
            )

        return out

    @staticmethod
    def _dedupe_and_cap(entities: list[Entity]) -> list[Entity]:
        """
        dedupe: remove if there are duplicates in the same chunk
        cap: top limit of enitities
        """
        seen: set[tuple[str, str, str, str | None]] = set()
        out: list[Entity] = []

        for e in entities:
            key = (e.text.strip().lower(), e.label, e.source, e.chunk_id)
            if key in seen:
                continue

            seen.add(key)
            out.append(e)

            if len(out) >= settings.MAX_ENTITIES:
                break

        return out

    def extract_entities(self, answer: str, sources: list[RetrievedChunk]) -> list[Entity]:
        """
        Extract entities from:
            - final answer (source="answer")
            - each retrieved chunk snippet (source="chunk", includes doc_id/page/chunk_id)
        """
        all_entities: list[Entity] = []

        # answer entities
        ans_text = (answer or "").strip()
        if ans_text:
            for ent in self._extract_from_text(ans_text):
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

        # chunk entities
        for ch in sources:
            chunk_text = (ch.text_snippet or "").strip()
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
                        doc_id=ch.doc_id,
                        page=ch.page,
                        chunk_id=ch.chunk_id,
                    )
                )

        return self._dedupe_and_cap(all_entities)


def default_ner_service() -> NerService:
    return NerService(settings.NER_MODEL_NAME)
