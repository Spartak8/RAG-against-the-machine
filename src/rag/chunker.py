"""Code and documentation chunking strategies for the vLLM codebase."""

import ast
from pathlib import Path
import re

from ..models import MinimalSource

SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}


class Chunk(MinimalSource):
    """A source span together with its raw text content."""

    text: str


def _compute_line_offsets(content: str) -> list[int]:
    """Return starting character index for each line plus total length."""
    offsets: list[int] = [0]
    for line in content.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _split_span(
    content: str,
    start: int,
    end: int,
    max_chunk_size: int,
    overlap: int = 200,
) -> list[tuple[int, int]]:
    """Split a character range into overlapping pieces <= max_chunk_size."""
    if max_chunk_size < 1:
        raise ValueError("Max Chunk Size must be greater than 0")
    spans: list[tuple[int, int]] = []
    safe_overlap = min(overlap, max_chunk_size // 4)
    pos = start

    while pos < end:
        chunk_end = min(pos + max_chunk_size, end)
        if chunk_end < end:
            newline_pos = content.rfind("\n", pos + max_chunk_size // 2,
                                        chunk_end)
            if newline_pos != -1:
                chunk_end = newline_pos + 1

        if content[pos:chunk_end].strip():
            spans.append((pos, chunk_end))

        if chunk_end >= end:
            break
        pos = max(pos + 1, chunk_end - safe_overlap)

    return spans


def chunk_python(
    file_path: str,
    content: str,
    max_chunk_size: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk Python code along AST class/function boundaries with fallback."""
    if not content.strip():
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        spans = _split_span(content, 0, len(content), max_chunk_size, overlap)
        return [
            Chunk(
                file_path=file_path,
                first_character_index=s,
                last_character_index=e,
                text=content[s:e],
            )
            for s, e in spans
        ]

    offsets = _compute_line_offsets(content)
    raw_spans: list[tuple[int, int]] = []
    cursor = 0

    for node in tree.body:
        if not isinstance(
            node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        if node.end_lineno is None:
            continue
        if node.decorator_list:
            start_idx = node.decorator_list[0].lineno - 1
        else:
            start_idx = node.lineno - 1
        end_idx = node.end_lineno
        if start_idx < 0 or end_idx >= len(offsets):
            continue

        start = offsets[start_idx]
        end = offsets[end_idx]

        if start > cursor and content[cursor:start].strip():
            raw_spans.append((cursor, start))
        raw_spans.append((start, end))
        cursor = end

    if cursor < len(content) and content[cursor:].strip():
        raw_spans.append((cursor, len(content)))

    if not raw_spans:
        raw_spans = [(0, len(content))]

    chunks: list[Chunk] = []
    for start, end in raw_spans:
        for s, e in _split_span(content, start, end, max_chunk_size, overlap):
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=s,
                    last_character_index=e,
                    text=content[s:e],
                )
            )
    return chunks


def chunk_markdown(
    file_path: str,
    content: str,
    max_chunk_size: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk Markdown or plain text files along heading boundaries."""
    if not content.strip():
        return []

    fenced_ranges: list[tuple[int, int]] = []
    for fence in re.finditer(r"(?m)^(`{3,}|~{3,}).*\n", content):
        if fenced_ranges and not fenced_ranges[-1][1]:
            fenced_ranges[-1] = (fenced_ranges[-1][0], fence.end())
        else:
            fenced_ranges.append((fence.start(), 0))
    if fenced_ranges and not fenced_ranges[-1][1]:
        fenced_ranges[-1] = (fenced_ranges[-1][0], len(content))

    def _inside_fence(pos: int) -> bool:
        return any(s <= pos < e for s, e in fenced_ranges)

    boundaries = [0]
    for match in re.finditer(r"(?m)^#{1,6}\s+", content):
        if match.start() > 0 and not _inside_fence(match.start()):
            boundaries.append(match.start())
    boundaries.append(len(content))

    merged: list[tuple[int, int]] = []
    for idx in range(len(boundaries) - 1):
        sec_start = boundaries[idx]
        sec_end = boundaries[idx + 1]
        if not content[sec_start:sec_end].strip():
            continue
        if merged and (sec_end - merged[-1][0]) <= max_chunk_size:
            if (merged[-1][1] - merged[-1][0]) <= max_chunk_size // 2:
                merged[-1] = (merged[-1][0], sec_end)
                continue
        merged.append((sec_start, sec_end))

    chunks: list[Chunk] = []
    for start, end in merged:
        for s, e in _split_span(content, start, end, max_chunk_size, overlap):
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=s,
                    last_character_index=e,
                    text=content[s:e],
                )
            )
    return chunks


def chunk_file(
    path: Path, max_chunk_size: int = 2000, overlap: int = 200
) -> list[Chunk]:
    """Read a supported file and dispatch it to the right chunker."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return []

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            content = handle.read()
    except OSError:
        return []

    rel_path = path.as_posix().removeprefix("./")

    if suffix == ".py":
        return chunk_python(rel_path, content, max_chunk_size, overlap)
    return chunk_markdown(rel_path, content, max_chunk_size, overlap)
