from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings


@dataclass(frozen=True)
class EmbedConfig:
    model_name: str
    batch_size: int
    normalize: bool


class EmbeddingService:
    """
    Singleton service (load this only once, optimization)
    """

    def __init__(self, cfg: EmbedConfig):
        self.cfg = cfg
        self._model: SentenceTransformer | None = None

    def load(self) -> None:
        if self._model is None:
            self._model = SentenceTransformer(self.cfg.model_name)

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            raise RuntimeError("Embedding model is not loaded. Call load() first.")

        return self._model

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """
        This returns float32 embeddings.
        Embeddings size: (N x D):
            - N number of chunks
            - D dimension
        """
        t0 = time.perf_counter()
        emb = self.model.encode(
            texts,
            batch_size=self.cfg.batch_size,
            normalize_embeddings=self.cfg.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        # ensure float32 for FAISS and disk size
        arr = np.asarray(emb, dtype=np.float32)
        _ = time.perf_counter() - t0

        return arr


def default_embedding_service() -> EmbeddingService:
    cfg = EmbedConfig(
        model_name=settings.EMBEDDING_MODEL_NAME,
        batch_size=settings.EMBEDDING_BATCH_SIZE,
        normalize=settings.EMBEDDING_NORMALIZE,
    )

    svc = EmbeddingService(cfg)

    return svc
