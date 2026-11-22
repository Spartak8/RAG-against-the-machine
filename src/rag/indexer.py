"""Build and persist an inverted BM25 index for the codebase."""

from collections import Counter, defaultdict
import math
from pathlib import Path
import pickle
import re

from tqdm import tqdm

from ..models import MinimalSource
from .chunker import (
    SUPPORTED_EXTENSIONS,
    _validate_chunk_size,
    _validate_overlap,
    chunk_file,
)

INDEX_FILENAME = "bm25_index.pkl"
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "its", "of", "on", "or",
    "that", "the", "this", "to", "use", "used", "using", "was", "what",
    "when", "where", "which", "with",
}
_CAMEL_RE = re.compile(r"([a-z0-9])([A-Z])")
_ACRONYM_RE = re.compile(r"([A-Z]+)([A-Z][a-z])")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Split text into code-aware lowercase tokens."""
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text):
        lower_full = raw.lower()
        if len(lower_full) > 1 and lower_full not in STOP_WORDS:
            tokens.append(lower_full)

        spaced = _ACRONYM_RE.sub(r"\1 \2", raw)
        spaced = _CAMEL_RE.sub(r"\1 \2", spaced).replace("_", " ")
        parts = spaced.lower().split()
        if len(parts) > 1:
            for part in parts:
                if len(part) > 1 and part not in STOP_WORDS:
                    tokens.append(part)
    return tokens


def _path_tokens(file_path: str) -> list[str]:
    """Extract informative tokens from a relative file path."""
    cleaned = file_path.replace("data/raw/", "")
    parts = cleaned.split("/", 1)
    relevant = parts[1] if len(parts) > 1 else cleaned
    return tokenize(relevant)


def build_index(
    raw_dir: str = "data/raw",
    output_dir: str = "data/processed",
    max_chunk_size: int = 2000,
    overlap: int = 200,
) -> int:
    """Chunk files in raw_dir and save a BM25 inverted index to output_dir."""
    _validate_chunk_size(max_chunk_size)
    _validate_overlap(overlap)

    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    files = sorted(
        p for p in raw_path.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    sources: list[MinimalSource] = []
    doc_lengths: list[int] = []
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for file_path in tqdm(files, desc="Indexing", unit="file"):
        chunks = chunk_file(
            file_path, max_chunk_size=max_chunk_size, overlap=overlap
        )
        for chunk in chunks:
            doc_id = len(sources)
            p_toks = _path_tokens(chunk.file_path)
            tokens = tokenize(chunk.text) + p_toks + p_toks
            if not tokens:
                continue

            sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )
            doc_lengths.append(len(tokens))
            for term, freq in Counter(tokens).items():
                postings[term].append((doc_id, freq))

    num_docs = len(sources)
    avgdl = (sum(doc_lengths) / num_docs) if num_docs > 0 else 0.0
    idf: dict[str, float] = {}
    for term, docs in postings.items():
        df = len(docs)
        idf[term] = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)

    index_payload = {
        "sources": [s.model_dump() for s in sources],
        "doc_lengths": doc_lengths,
        "avgdl": avgdl,
        "postings": dict(postings),
        "idf": idf,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    with (out_path / INDEX_FILENAME).open("wb") as handle:
        pickle.dump(index_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return num_docs
