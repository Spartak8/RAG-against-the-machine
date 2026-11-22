"""BM25 retrieval over the persisted inverted index with caching."""

from collections import defaultdict
import json
from pathlib import Path
import pickle
from typing import Any

from ..models import MinimalSource
from .indexer import INDEX_FILENAME, tokenize

QUERY_CACHE_FILENAME = "query_cache.json"
MAX_CACHE_ENTRIES = 512

_INDEX_CACHE: dict[tuple[str, float], dict[str, Any]] = {}


def _load_index_data(index_file: Path) -> tuple[dict[str, Any], float]:
    """Load index payload using an mtime-keyed in-memory cache."""
    resolved = str(index_file.resolve())
    mtime = index_file.stat().st_mtime
    cache_key = (resolved, mtime)

    cached = _INDEX_CACHE.get(cache_key)
    if cached is not None:
        return cached, mtime

    with index_file.open("rb") as handle:
        data: dict[str, Any] = pickle.load(handle)

    stale_keys = [k for k in _INDEX_CACHE if k[0] == resolved]
    for key in stale_keys:
        del _INDEX_CACHE[key]
    _INDEX_CACHE[cache_key] = data
    return data, mtime


class BM25Retriever:
    """Fast Okapi BM25 retriever backed by a cached inverted index."""

    def __init__(
        self,
        index_dir: str = "data/processed",
        k1: float = 1.5,
        b: float = 0.65,
        use_cache: bool = True,
    ) -> None:
        """Load the persisted BM25 index and optional disk query cache."""
        index_file = Path(index_dir) / INDEX_FILENAME
        if not index_file.exists():
            raise FileNotFoundError(
                f"Index not found at {index_file}. Run 'index' first."
            )

        data, mtime = _load_index_data(index_file)

        self.index_dir = Path(index_dir)
        self.index_mtime = mtime
        self.sources: list[MinimalSource] = [
            MinimalSource.model_validate(item) for item in data["sources"]
        ]
        self.doc_lengths: list[int] = data["doc_lengths"]
        self.avgdl: float = float(data["avgdl"])
        self.postings: dict[str, list[tuple[int, int]]] = data["postings"]
        self.idf: dict[str, float] = data["idf"]
        self.k1 = k1
        self.b = b
        self.use_cache = use_cache
        self._cache_file = self.index_dir / QUERY_CACHE_FILENAME
        self._query_cache: dict[str, list[dict[str, Any]]] = {}
        if self.use_cache:
            self._load_query_cache()

    def _load_query_cache(self) -> None:
        """Load persisted query results if they match the current index."""
        if not self._cache_file.exists():
            return
        try:
            with self._cache_file.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            if (
                isinstance(raw, dict)
                and raw.get("index_mtime") == self.index_mtime
                and isinstance(raw.get("entries"), dict)
            ):
                self._query_cache = raw["entries"]
        except (OSError, json.JSONDecodeError, ValueError):
            self._query_cache = {}

    def _save_query_cache(self) -> None:
        """Persist query cache to disk without failing on I/O errors."""
        payload = {
            "index_mtime": self.index_mtime,
            "entries": self._query_cache,
        }
        try:
            with self._cache_file.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
        except OSError:
            pass

    def search(self, query: str, k: int = 5) -> list[MinimalSource]:
        """Return top-k MinimalSource objects ranked by BM25 score."""
        if not isinstance(query, str) or isinstance(k, bool):
            return []
        if not isinstance(k, int) or k <= 0:
            return []
        if not query.strip() or not self.sources:
            return []

        cache_key = f"{k}:{query.strip()}"
        if self.use_cache and cache_key in self._query_cache:
            return [
                MinimalSource.model_validate(item)
                for item in self._query_cache[cache_key]
            ]

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
        results = [self.sources[doc_id] for doc_id in top_ids]

        if self.use_cache:
            if len(self._query_cache) >= MAX_CACHE_ENTRIES:
                oldest = next(iter(self._query_cache))
                del self._query_cache[oldest]
            self._query_cache[cache_key] = [s.model_dump() for s in results]
            self._save_query_cache()

        return results
