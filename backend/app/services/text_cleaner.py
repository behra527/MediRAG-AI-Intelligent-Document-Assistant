
"""
MediRAG AI - Clinical Text Cleaning and Preprocessing.

Conservative text hygiene for extracted clinical guidelines.

Compatible with:
    main.py
    compare_parsers.py
    pdf_extractor.py

Python 3.13
"""

from __future__ import annotations

import logging
import re
import unicodedata

from collections import Counter
from dataclasses import dataclass, field


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------
# Regular expressions
# ---------------------------------------------------------

PAGE_RE = re.compile(
    r"^(?:page\s+)?\d+\s*(?:/|of)\s*\d+$",
    re.IGNORECASE,
)

PAGE_NUMBER_RE = re.compile(
    r"^(?:page\s+)\d+$",
    re.IGNORECASE,
)

HYPHEN_RE = re.compile(
    r"(?<=[A-Za-z])[-\u00ad]\n(?=[a-z])"
)

WHITESPACE_RE = re.compile(r"[ \t]+")

MULTIPLE_NEWLINES_RE = re.compile(r"\n{3,}")

TABLE_START = "[TABLE]"
TABLE_END = "[/TABLE]"


# ---------------------------------------------------------
# Data models
# ---------------------------------------------------------

@dataclass
class CleanResult:
    """
    Result of cleaning a collection of PDF pages.

    Existing fields are retained for compatibility.
    """

    pages: list[str]

    removed_lines: int = 0

    removed_examples: list[str] = field(
        default_factory=list
    )

    hyphen_repairs: int = 0

    warnings: list[str] = field(
        default_factory=list
    )

    @property
    def text(self) -> str:
        """
        Return all cleaned pages.

        Page boundaries remain separated by blank lines.
        """
        return "\n\n".join(
            page for page in self.pages if page.strip()
        )


# ---------------------------------------------------------
# Text normalization
# ---------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Normalize common PDF typography without changing
    medical symbols, numerical values, or units.

    Avoid NFKC normalization because it can alter
    meaningful scientific notation.
    """

    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "–",
        "\u2014": "—",
        "\u2212": "−",
    }

    for original, replacement in replacements.items():
        text = text.replace(original, replacement)

    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """
    Remove excessive spaces while preserving
    paragraph and table boundaries.
    """

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = WHITESPACE_RE.sub(" ", text)

    text = re.sub(
        r" *\n *",
        "\n",
        text,
    )

    text = MULTIPLE_NEWLINES_RE.sub(
        "\n\n",
        text,
    )

    return text.strip()


# ---------------------------------------------------------
# Header/footer detection
# ---------------------------------------------------------

def _key(value: str) -> str:
    """
    Normalize recurring header/footer candidates.

    Numeric page identifiers are replaced so that
    'Guideline | 12' and 'Guideline | 13' match.
    """

    value = value.casefold()

    value = re.sub(
        r"\d+",
        "#",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip(" -|:")


def _edge_indices(
    lines: list[str],
    margin_lines: int,
) -> set[int]:
    """
    Return indices of the first and last meaningful
    lines of a page.
    """

    nonempty = [
        index
        for index, line in enumerate(lines)
        if line.strip()
    ]

    if not nonempty:
        return set()

    edge_count = min(
        margin_lines,
        max(1, len(nonempty) // 5),
    )

    return set(
        nonempty[:edge_count]
        + nonempty[-edge_count:]
    )


def _is_table_line(line: str) -> bool:
    """
    Identify table markers or pipe-delimited rows.
    """

    stripped = line.strip()

    return (
        stripped in {TABLE_START, TABLE_END}
        or (
            "|" in stripped
            and len(stripped.split("|")) >= 2
        )
    )


def _table_line_indices(
    lines: list[str],
) -> set[int]:
    """
    Protect every line inside a [TABLE] block.

    This prevents table headers, medication names,
    dosage values, and repeated clinical entries
    from being removed as page noise.
    """

    protected: set[int] = set()

    inside_table = False

    for index, line in enumerate(lines):
        stripped = line.strip()

        if stripped == TABLE_START:
            inside_table = True

        if inside_table or _is_table_line(stripped):
            protected.add(index)

        if stripped == TABLE_END:
            inside_table = False

    return protected


def detect_repeated_noise(
    pages: list[str],
    margin_lines: int = 3,
    min_repeated_pages: int = 3,
) -> set[str]:
    """
    Detect recurring text near page boundaries.

    Only margin candidates are considered.

    A repeated line appearing inside the clinical
    body is not automatically removed.
    """

    if margin_lines < 1:
        raise ValueError(
            "margin_lines must be at least 1"
        )

    if min_repeated_pages < 1:
        raise ValueError(
            "min_repeated_pages must be at least 1"
        )

    candidates: Counter[str] = Counter()

    for page in pages:
        lines = page.splitlines()

        edges = _edge_indices(
            lines,
            margin_lines,
        )

        protected = _table_line_indices(lines)

        page_candidates: set[str] = set()

        for index in edges:
            if index in protected:
                continue

            line = lines[index].strip()

            if not line:
                continue

            if len(line) > 110:
                continue

            page_candidates.add(_key(line))

        candidates.update(page_candidates)

    threshold = max(
        min_repeated_pages,
        (len(pages) + 1) // 2,
    )

    return {
        key
        for key, count in candidates.items()
        if count >= threshold
    }


# ---------------------------------------------------------
# Hyphenation repair
# ---------------------------------------------------------

def repair_hyphenation(
    text: str,
) -> tuple[str, int]:
    """
    Repair words broken across PDF lines.

    Example:
        hypo-
        tension

    Result:
        hypotension

    Conservative rules:
        - Only join alphabetic word fragments.
        - Never modify table blocks.
        - Never join across paragraph boundaries.
        - Preserve numerical ranges and dosage units.

    Some legitimate hyphenated compounds may still
    need validation against the source PDF.
    """

    repaired_count = 0
    output: list[str] = []
    buffer: list[str] = []

    def flush_buffer() -> None:
        nonlocal repaired_count

        if not buffer:
            return

        segment = "\n".join(buffer)

        segment, count = HYPHEN_RE.subn(
            "",
            segment,
        )

        repaired_count += count

        output.append(segment)

        buffer.clear()

    inside_table = False

    for line in text.split("\n"):
        stripped = line.strip()

        if stripped == TABLE_START:
            flush_buffer()
            inside_table = True
            output.append(line)
            continue

        if stripped == TABLE_END:
            flush_buffer()
            inside_table = False
            output.append(line)
            continue

        if inside_table or _is_table_line(line):
            flush_buffer()
            output.append(line)
            continue

        if not stripped:
            flush_buffer()
            output.append("")
            continue

        buffer.append(line)

    flush_buffer()

    return "\n".join(output), repaired_count


# ---------------------------------------------------------
# Main cleaning pipeline
# ---------------------------------------------------------

def clean_pages(
    pages: list[str],
    *,
    margin_lines: int = 3,
    min_repeated_pages: int = 3,
) -> CleanResult:
    """
    Clean extracted clinical PDF pages.

    Pipeline:
        1. Normalize Unicode.
        2. Identify recurring page-margin noise.
        3. Remove repeated headers and footers.
        4. Repair broken word hyphenation.
        5. Normalize whitespace.
        6. Preserve tables and page boundaries.

    Returns:
        CleanResult with cleaned pages and diagnostics.
    """

    if not isinstance(pages, list):
        raise TypeError(
            "pages must be a list of strings"
        )

    if any(not isinstance(page, str) for page in pages):
        raise TypeError(
            "Every page must contain text"
        )

    if margin_lines < 1:
        raise ValueError(
            "margin_lines must be at least 1"
        )

    if min_repeated_pages < 1:
        raise ValueError(
            "min_repeated_pages must be at least 1"
        )

    normalized_pages = [
        normalize_unicode(page)
        for page in pages
    ]

    repeated_noise = detect_repeated_noise(
        normalized_pages,
        margin_lines=margin_lines,
        min_repeated_pages=min_repeated_pages,
    )

    result = CleanResult(pages=[])

    for page_number, page in enumerate(
        normalized_pages,
        start=1,
    ):
        lines = page.splitlines()

        edges = _edge_indices(
            lines,
            margin_lines,
        )

        protected = _table_line_indices(lines)

        kept_lines: list[str] = []

        for index, line in enumerate(lines):
            stripped = line.strip()

            should_remove = (
                index in edges
                and index not in protected
                and bool(stripped)
                and (
                    PAGE_RE.fullmatch(stripped) is not None
                    or PAGE_NUMBER_RE.fullmatch(stripped)
                    is not None
                    or (
                        len(stripped) <= 110
                        and _key(stripped) in repeated_noise
                    )
                )
            )

            if should_remove:
                result.removed_lines += 1

                if len(result.removed_examples) < 12:
                    result.removed_examples.append(
                        stripped
                    )

                continue

            kept_lines.append(line)

        cleaned = "\n".join(kept_lines)

        cleaned, repair_count = repair_hyphenation(
            cleaned
        )

        result.hyphen_repairs += repair_count

        cleaned = normalize_whitespace(cleaned)

        result.pages.append(cleaned)

        if not cleaned:
            result.warnings.append(
                f"Page {page_number}: no text after cleaning"
            )

    LOGGER.info(
        "Clinical text cleaning completed: "
        "pages=%s removed_lines=%s hyphen_repairs=%s",
        len(result.pages),
        result.removed_lines,
        result.hyphen_repairs,
    )

    return result
