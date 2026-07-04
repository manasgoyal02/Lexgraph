import re
from typing import Pattern

# Built-in heading styles seen in legal documents.
DEFAULT_HEADER_PATTERNS: tuple[str, ...] = (
    r"(?m)^\s*\[\[HDR\]\]\s*(?P<number>(?:ARTICLE|Article)\s+[IVXLCM\d]+|Section\s+\d+(?:\.\d+)*[A-Za-z]?|Article\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*|[A-Z]|\([a-zivxlcdm]+\))[\s:\.-]+",
    r"(?m)^\s*\[\[HDR\]\]\s*(?P<number>(?:ARTICLE|Article)\s+[IVXLCM\d]+|Section\s+\d+(?:\.\d+)*[A-Za-z]?|Article\s+\d+(?:\.\d+)*|\d+(?:\.\d+)*|[A-Z]|\([a-zivxlcdm]+\))\b",
    r"(?m)^\s*(?P<number>\d+(?:\.\d+)*)[\.)]?\s+",  # 1, 1.2, 3.1.4
    r"(?m)^\s*(?P<number>\([a-z]\)|\([ivxlcdm]+\))\s+",  # (a), (iv)
    r"(?m)^\s*(?P<number>[A-Z])[\.)]\s+",  # A. / B)
    r"(?m)^\s*(?P<number>(?:Article|Section)\s+\d+(?:\.\d+)*)\s+",  # Article 1, Section 2.3
    r"(?m)^\s*(?P<number>ARTICLE\s+[IVXLCM]+)\b[\s:\.-]*",  # ARTICLE I, ARTICLE IV
    r"(?m)^\s*(?P<number>Article\s+[IVXLCM]+)\b[\s:\.-]*",  # Article I, Article IV
    r"(?m)^\s*(?P<number>Section\s+\d+(?:\.\d+)*[A-Za-z]?)\b[\s:\.-]*",  # Section 1.01, Section 2A
)

ClauseRecord = dict[str, str]


def _normalize_clause_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _compile_patterns(header_patterns: list[str] | tuple[str, ...] | None) -> list[Pattern[str]]:
    patterns = header_patterns or list(DEFAULT_HEADER_PATTERNS)
    return [re.compile(pattern) for pattern in patterns]


def _collect_header_matches(text: str, compiled_patterns: list[Pattern[str]]) -> list[tuple[int, int, str]]:
    raw_matches: list[tuple[int, int, str]] = []
    for pattern in compiled_patterns:
        for match in pattern.finditer(text):
            number = (match.groupdict().get("number") or "").strip()
            raw_matches.append((match.start(), match.end(), number))

    if not raw_matches:
        return []

    # Merge overlaps by keeping the broadest match for same start.
    by_start: dict[int, tuple[int, int, str]] = {}
    for start, end, number in raw_matches:
        current = by_start.get(start)
        if current is None or end > current[1]:
            by_start[start] = (start, end, number)

    merged = sorted(by_start.values(), key=lambda x: x[0])

    # Drop nested/overlapping starts that are not valid new boundaries.
    filtered: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, number in merged:
        if start < last_end:
            continue
        filtered.append((start, end, number))
        last_end = max(last_end, end)
    return filtered


def split_into_clauses(
    text: str,
    *,
    header_patterns: list[str] | tuple[str, ...] | None = None,
    collapse_whitespace: bool = True,
    structured: bool = False,
) -> list[str] | list[ClauseRecord]:
    """Split a contract-like body of text into clauses.

    Args:
        text: Raw document text.
        header_patterns: Optional regex patterns that detect clause starts.
            Patterns should usually include a named group `number`.
        collapse_whitespace: If True, normalize whitespace per clause.
        structured: If True, return [{"number": ..., "text": ...}] records.

    Returns:
        A list of clause strings by default, or structured records when
        `structured=True`.
    """
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    compiled = _compile_patterns(header_patterns)
    matches = _collect_header_matches(normalized, compiled)

    if matches:
        clause_chunks: list[str] = []
        structured_chunks: list[ClauseRecord] = []

        for i, match in enumerate(matches):
            start = match[0]
            end = matches[i + 1][0] if i + 1 < len(matches) else len(normalized)
            raw_chunk = normalized[start:end].strip()
            if not raw_chunk:
                continue

            chunk = raw_chunk.replace("[[HDR]] ", "").strip()
            rendered = _normalize_clause_whitespace(chunk) if collapse_whitespace else chunk
            clause_chunks.append(rendered)

            if structured:
                number = (match[2] or "").strip()
                body_raw = raw_chunk[match[1] - match[0] :].strip()
                body = body_raw.replace("[[HDR]] ", "").strip()
                body_rendered = _normalize_clause_whitespace(body) if collapse_whitespace else body
                structured_chunks.append({"number": number, "text": body_rendered})

        return structured_chunks if structured else clause_chunks

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]
    if collapse_whitespace:
        paragraphs = [_normalize_clause_whitespace(p) for p in paragraphs]

    if structured:
        return [{"number": "", "text": p} for p in paragraphs]

    return paragraphs
