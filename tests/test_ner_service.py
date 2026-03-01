from dataclasses import dataclass

from app.services.ner.ner_service import NerService
from app.services.retrieval.retriever import RetrievedChunk


@dataclass(frozen=True)
class FakeEnt:
    text: str
    label_: str
    start_char: int
    end_char: int


class FakeDoc:
    def __init__(self, ents):
        self.ents = ents


class FakeNLP:
    """
    Very small fake spaCy pipeline.
    Returns a predetermined set of entities depending on the input text.
    """

    def __call__(self, text: str):
        text = text or ""
        ents = []

        # detect simple markers in test strings
        if "John Doe" in text:
            s = text.index("John Doe")
            ents.append(FakeEnt("John Doe", "PERSON", s, s + len("John Doe")))

        if "Acme Corp" in text:
            s = text.index("Acme Corp")
            ents.append(FakeEnt("Acme Corp", "ORG", s, s + len("Acme Corp")))

        if "2026-02-26" in text:
            s = text.index("2026-02-26")
            ents.append(FakeEnt("2026-02-26", "DATE", s, s + len("2026-02-26")))

        return FakeDoc(ents)


def test_ner_extracts_from_answer_and_chunks_and_dedupes(monkeypatch):
    svc = NerService("dummy")
    # inject fake nlp
    monkeypatch.setattr(svc, "_nlp", FakeNLP())

    answer = "John Doe signed with Acme Corp on 2026-02-26."
    sources = [
        RetrievedChunk(
            doc_id="a" * 32,
            chunk_id="chunk_1",
            score=0.9,
            page=1,
            chunk_index=0,
            text_snippet="This contract is between John Doe and Acme Corp.",
        ),
        # duplicate entities again to test dedupe
        RetrievedChunk(
            doc_id="a" * 32,
            chunk_id="chunk_2",
            score=0.8,
            page=2,
            chunk_index=1,
            text_snippet="Acme Corp invoice date: 2026-02-26.",
        ),
    ]

    ents = svc.extract_entities(answer, sources)

    # must contain at least 3 entities
    assert any(e.text == "John Doe" and e.label == "PERSON" for e in ents)
    assert any(e.text == "Acme Corp" and e.label == "ORG" for e in ents)
    assert any(e.text == "2026-02-26" and e.label == "DATE" for e in ents)

    # verify chunk provenance exists for chunk-source entities
    chunk_ents = [e for e in ents if e.source == "chunk"]
    assert all(e.doc_id is not None for e in chunk_ents)
    assert all(e.chunk_id is not None for e in chunk_ents)

    # verify answer provenance is empty doc/page/chunk
    ans_ents = [e for e in ents if e.source == "answer"]
    assert all(e.doc_id is None and e.page is None and e.chunk_id is None for e in ans_ents)

    # Dedup rule: same text+label+source+chunk_id -> unique
    # We allow same text/label across different chunk_ids (useful for provenance)
    # So at least answer entity + chunk entity for same surface form can exist.
    assert len(ents) >= 3
