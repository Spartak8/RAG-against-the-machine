"""BM25 retrieval over the persisted inverted index."""

from collections import defaultdict
from pathlib import Path
import pickle
from typing import Any

from ..models import MinimalSource
from .indexer import INDEX_FILENAME, tokenize


class BM25Retriever:
    """Fast Okapi BM25 retriever backed by an inverted index."""

    def __init__(
        self,
        index_dir: str = "data/processed",
        k1: float = 1.5,
        b: float = 0.65,
    ) -> None:
        """Load the persisted BM25 index from disk."""
        index_file = Path(index_dir) / INDEX_FILENAME
        if not index_file.exists():
            raise FileNotFoundError(
                f"Index not found at {index_file}. Run 'index' first."
            )

        with index_file.open("rb") as handle:
            data: dict[str, Any] = pickle.load(handle)

        self.sources: list[MinimalSource] = [
            MinimalSource.model_validate(item) for item in data["sources"]
        ]
        self.doc_lengths: list[int] = data["doc_lengths"]
        self.avgdl: float = float(data["avgdl"])
        self.postings: dict[str, list[tuple[int, int]]] = data["postings"]
        self.idf: dict[str, float] = data["idf"]
        self.k1 = k1
        self.b = b

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:
        """Return top-k MinimalSource objects ranked by BM25 score."""
        if not isinstance(query, str) or isinstance(k, bool):
            return []
        if not isinstance(k, int) or k <= 0:
            return []
        if not query.strip() or not self.sources:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        avgdl = self.avgdl if self.avgdl > 0 else 1.0

        for term in query_tokens:
            term_postings = self.postings.get(term)
            if not term_postings:
                continue
            term_idf = self.idf[term]
            for doc_id, tf in term_postings:
                dl = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))
                scores[doc_id] += term_idf * (tf * (self.k1 + 1.0)) / denom

        if not scores:
            return []

        top_ids = sorted(
            scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True
        )[:k]
        return [self.sources[doc_id] for doc_id in top_ids]
