"""Code and documentation chunking strategies for the vLLM codebase."""

import ast
import os
import re
from pathlib import Path

from ..models import MinimalSource

SUPPORTED_EXTENSIONS = {".py", ".md", ".txt"}
MAX_ALLOWED_CHUNK_SIZE = 2000

_NEWLINE_RE = re.compile(r"\r\n|\r|\n")
_FENCE_RE = re.compile(r" {0,3}(`{3,}|~{3,})(.*)")
_ATX_HEADING_RE = re.compile(r" {0,3}#{1,6}(?:[ \t]+|$)")


class Chunk(MinimalSource):
    """A source span together with its raw text content."""

    text: str


def _validate_chunk_size(max_chunk_size: int) -> None:
    """Raise ValueError unless chunk size is a valid moulinette-safe int."""
    if isinstance(max_chunk_size, bool) or not isinstance(
        max_chunk_size, int
    ):
        raise ValueError(
            "max_chunk_size must be an int, "
            f"got {type(max_chunk_size).__name__}"
        )
    if not 1 <= max_chunk_size <= MAX_ALLOWED_CHUNK_SIZE:
        raise ValueError(
            f"max_chunk_size must be between 1 and {MAX_ALLOWED_CHUNK_SIZE}, "
            f"got {max_chunk_size}"
        )


def _validate_overlap(overlap: int) -> None:
    """Raise ValueError unless overlap is an int (excluding bool)."""
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise ValueError(
            f"overlap must be an int, got {type(overlap).__name__}"
        )


def _compute_line_offsets(content: str) -> list[int]:
    """Return the starting character index of each line plus total length."""
    offsets: list[int] = [0]
    offsets.extend(match.end() for match in _NEWLINE_RE.finditer(content))
    if offsets[-1] != len(content):
        offsets.append(len(content))
    return offsets


def _fenced_ranges(content: str) -> list[tuple[int, int]]:
    """Return the character ranges of fenced code blocks (CommonMark rules)."""
    ranges: list[tuple[int, int]] = []
    offsets = _compute_line_offsets(content)
    open_start = -1
    open_char = ""
    open_len = 0

    for idx in range(len(offsets) - 1):
        line_start, line_end = offsets[idx], offsets[idx + 1]
        line = content[line_start:line_end].rstrip("\r\n")
        match = _FENCE_RE.fullmatch(line)
        if match is None:
            continue
        marks, info = match.groups()
        if open_start < 0:
            if marks[0] == "`" and "`" in info:
                continue
            open_start, open_char, open_len = line_start, marks[0], len(marks)
        elif (
            marks[0] == open_char
            and len(marks) >= open_len
            and not info.strip()
        ):
            ranges.append((open_start, line_end))
            open_start = -1

    if open_start >= 0:
        ranges.append((open_start, len(content)))
    return ranges


def _find_last_line_break_end(content: str, low: int, high: int) -> int:
    """Return end index of the last line break in [low, high), or -1."""
    for idx in range(high - 1, low - 1, -1):
        ch = content[idx]
        if ch == "\n":
            return idx + 1
        if ch == "\r":
            if idx + 1 >= len(content) or content[idx + 1] != "\n":
                return idx + 1
    return -1


def _strip_trailing_line_break_end(
    content: str, start: int, end: int
) -> int:
    """Return index just before a trailing line terminator at ``end``."""
    if end > start and content[end - 1] == "\n":
        if end - 2 >= start and content[end - 2] == "\r":
            return end - 2
        return end - 1
    if end > start and content[end - 1] == "\r":
        return end - 1
    return end


def _split_span(
    content: str,
    start: int,
    end: int,
    max_chunk_size: int,
    overlap: int = 200,
) -> list[tuple[int, int]]:
    """Split a character range into overlapping pieces <= max_chunk_size."""
    _validate_chunk_size(max_chunk_size)
    _validate_overlap(overlap)
    spans: list[tuple[int, int]] = []
    safe_overlap = max(0, min(overlap, max_chunk_size // 4))
    pos = start

    while pos < end:
        chunk_end = min(pos + max_chunk_size, end)
        if chunk_end < end:
            break_end = _find_last_line_break_end(
                content, pos + max_chunk_size // 2, chunk_end
            )
            if break_end != -1:
                chunk_end = break_end
            elif (
                chunk_end - 1 > pos
                and content[chunk_end - 1] == "\r"
                and content[chunk_end] == "\n"
            ):
                chunk_end -= 1

        if content[pos:chunk_end].strip():
            spans.append((pos, chunk_end))

        if chunk_end >= end:
            break

        next_pos = chunk_end - safe_overlap
        if safe_overlap:
            search_end = _strip_trailing_line_break_end(
                content, pos, chunk_end
            )
            match = _NEWLINE_RE.search(content, next_pos, search_end)
            if match is not None:
                next_pos = match.end()
            elif (
                next_pos > pos
                and content[next_pos - 1] == "\r"
                and content[next_pos] == "\n"
            ):
                if next_pos + 1 < chunk_end:
                    next_pos += 1
                elif next_pos - 1 > pos:
                    next_pos -= 1

        pos = max(pos + 1, next_pos)

    return spans


def _split_ranges(
    content: str,
    ranges: list[tuple[int, int]],
    max_chunk_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Apply ``_split_span`` to every range and join the results."""
    spans: list[tuple[int, int]] = []
    for start, end in ranges:
        spans.extend(_split_span(content, start, end, max_chunk_size, overlap))
    return spans


def _make_chunks(
    file_path: str, content: str, spans: list[tuple[int, int]]
) -> list[Chunk]:
    """Build Chunk objects (with their text) for the given spans."""
    return [
        Chunk(
            file_path=file_path,
            first_character_index=first,
            last_character_index=last,
            text=content[first:last],
        )
        for first, last in spans
    ]


def chunk_python(
    file_path: str,
    content: str,
    max_chunk_size: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk Python code along AST class/function boundaries with fallback."""
    _validate_chunk_size(max_chunk_size)
    _validate_overlap(overlap)
    if not content.strip():
        return []

    try:
        tree = ast.parse(content)
    except (SyntaxError, ValueError, RecursionError):
        spans = _split_span(content, 0, len(content), max_chunk_size, overlap)
        return _make_chunks(file_path, content, spans)

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

        if start > cursor:
            if content[cursor:start].strip():
                raw_spans.append((cursor, start))
            elif raw_spans and (start - raw_spans[-1][0]) <= max_chunk_size:
                raw_spans[-1] = (raw_spans[-1][0], start)
            elif not raw_spans and (end - cursor) <= max_chunk_size:
                start = cursor

        raw_spans.append((start, end))
        cursor = end

    if cursor < len(content):
        if content[cursor:].strip():
            raw_spans.append((cursor, len(content)))
        elif raw_spans and (len(content) - raw_spans[-1][0]) <= max_chunk_size:
            raw_spans[-1] = (raw_spans[-1][0], len(content))

    if not raw_spans:
        raw_spans = [(0, len(content))]

    spans = _split_ranges(content, raw_spans, max_chunk_size, overlap)
    return _make_chunks(file_path, content, spans)


def chunk_markdown(
    file_path: str,
    content: str,
    max_chunk_size: int = 2000,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk Markdown or plain text files along heading boundaries."""
    _validate_chunk_size(max_chunk_size)
    _validate_overlap(overlap)
    if not content.strip():
        return []

    fenced = _fenced_ranges(content)
    offsets = _compute_line_offsets(content)

    boundaries = [0]
    for idx in range(len(offsets) - 1):
        line_start, line_end = offsets[idx], offsets[idx + 1]
        if line_start == 0:
            continue
        if any(s <= line_start < e for s, e in fenced):
            continue
        line = content[line_start:line_end].rstrip("\r\n")
        if _ATX_HEADING_RE.match(line):
            boundaries.append(line_start)
    boundaries.append(len(content))

    merged: list[tuple[int, int]] = []
    pending_ws_start: int | None = None
    for idx in range(len(boundaries) - 1):
        sec_start = boundaries[idx]
        sec_end = boundaries[idx + 1]
        if not content[sec_start:sec_end].strip():
            if not merged and pending_ws_start is None:
                pending_ws_start = sec_start
            continue
        if pending_ws_start is not None:
            if (sec_end - pending_ws_start) <= max_chunk_size:
                sec_start = pending_ws_start
            pending_ws_start = None
        if merged and (sec_end - merged[-1][0]) <= max_chunk_size:
            if (merged[-1][1] - merged[-1][0]) <= max_chunk_size // 2:
                merged[-1] = (merged[-1][0], sec_end)
                continue
        merged.append((sec_start, sec_end))

    spans = _split_ranges(content, merged, max_chunk_size, overlap)
    return _make_chunks(file_path, content, spans)


def _corpus_path(path: Path) -> str:
    """Return ``path`` relative to the working directory, in POSIX form."""
    cwd = Path.cwd()
    logical_path = Path(os.path.abspath(path))
    logical_cwd = Path(os.path.abspath(cwd))
    try:
        return logical_path.relative_to(logical_cwd).as_posix()
    except ValueError:
        pass

    try:
        return path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def chunk_file(
    path: Path, max_chunk_size: int = 2000, overlap: int = 200
) -> list[Chunk]:
    """Read a supported file and dispatch it to the right chunker."""
    _validate_chunk_size(max_chunk_size)
    _validate_overlap(overlap)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        return []

    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            content = handle.read()
    except OSError:
        return []

    rel_path = _corpus_path(path)

    if suffix == ".py":
        return chunk_python(rel_path, content, max_chunk_size, overlap)
    return chunk_markdown(rel_path, content, max_chunk_size, overlap)
