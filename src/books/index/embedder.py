"""Text embedders.

The :class:`Embedder` Protocol describes the minimal interface that the rest
of the indexing pipeline depends on. The default implementation is
:class:`SentenceTransformersEmbedder`, which loads a HuggingFace model
locally — no API key required. Swap in another implementation by writing a
new class with the same shape and updating :func:`make_embedder`.
"""

from typing import Protocol


class Embedder(Protocol):
    """Minimum interface for an embedding backend.

    ``name`` should identify the model (e.g. ``"BAAI/bge-small-en-v1.5"``);
    ``dim`` is the output vector dimension; ``embed`` takes a list of texts
    and returns a list of equal length where each entry is the unit-normalised
    embedding vector.
    """

    name: str
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformersEmbedder:
    """Local embedder backed by `sentence-transformers`_.

    Loads the named model on construction (downloads from HuggingFace on
    first use). Embeddings are L2-normalised so cosine and dot-product give
    the same ranking.

    .. _sentence-transformers: https://www.sbert.net/
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        # Heavy import deferred to here so that simply importing this module
        # (e.g. for type checking) does not pull in torch.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)
        self.name = model_name
        # `get_embedding_dimension` is the new spelling; older releases use
        # `get_sentence_embedding_dimension`. Probe for whichever exists.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode ``texts`` and return them as plain Python float lists."""
        if not texts:
            return []
        embs = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()


def make_embedder(model: str, device: str) -> Embedder:
    """Construct the configured default embedder. Single-line indirection so
    callers don't have to know which implementation we ship."""
    return SentenceTransformersEmbedder(model, device=device)
