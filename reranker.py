"""
Cross-encoder reranker — stage 2 of retrieval.

ChromaDB's bi-encoder embeds query and documents separately, which is fast but
lossy. The cross-encoder reads each (query, document) pair jointly and produces
a much more accurate relevance score. We fetch a wide candidate set from
ChromaDB, then keep only the top-scoring examples after reranking.

The model (~90MB) is downloaded from HuggingFace on first use and cached.
"""

import config

_model = None


def get_reranker():
    """Lazily load the cross-encoder model (cached after first call)."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(config.RERANKER_MODEL)
    return _model


def rerank(query: str, documents: list, top_k: int) -> list:
    """Return the top_k documents most relevant to query, best first."""
    if len(documents) <= 1:
        return documents[:top_k]
    model = get_reranker()
    scores = model.predict([(query, doc) for doc in documents])
    ranked = sorted(zip(documents, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, _ in ranked[:top_k]]
