
"""Layout-aware PDF extraction for MediRAG AI."""

from __future__ import annotations

import io
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Literal

import fitz
import pdfplumber

LOGGER = logging.getLogger(__name__)

ParserName = Literal["pymupdf", "pdfplumber"]


@dataclass
class PageExtraction:
    page: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    columns_detected: int = 1
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    parser: str
    pages: list[PageExtraction]
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize_cell(value: object) -> str:
    """Normalize table cells without losing medical units."""
    if value is None:
        return ""
    return " ".join(str(value).split())


def _render_table(rows: list[list[str]]) -> str:
    """Render structured table rows for downstream RAG."""
    return "\n".join(
        " | ".join(_normalize_cell(cell) for cell in row)
        for row in rows
    )


def _intersects(
    box: tuple[float, float, float, float],
    table_boxes: list[tuple[float, float, float, float]],
) -> bool:
    """Check whether the center of a text block is inside a table."""
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2

    return any(
        tx0 <= cx <= tx1 and ty0 <= cy <= ty1
        for tx0, ty0, tx1, ty1 in table_boxes
    )


def _column_groups(
    blocks: list[tuple[float, float, float, float, str]],
    width: float,
) -> tuple[list[str], int]:
    """Separate left and right columns before assembling text."""
    midpoint = width / 2
    tolerance = width * 0.035

    left = []
    right = []
    spanning = []

    for block in blocks:
        x0, _, x1, _, _ = block

        if x1 <= midpoint + tolerance:
            left.append(block)
        elif x0 >= midpoint - tolerance:
            right.append(block)
        else:
            spanning.append(block)

    left_chars = sum(len(block[4]) for block in left)
    right_chars = sum(len(block[4]) for block in right)

    if left_chars >= 120 and right_chars >= 120:
        # Column-wise reading prevents horizontal sentence merging.
        ordered = (
            sorted(spanning, key=lambda b: (b[1], b[0]))
            + sorted(left, key=lambda b: (b[1], b[0]))
            + sorted(right, key=lambda b: (b[1], b[0]))
        )
        return [block[4] for block in ordered], 2

    ordered = sorted(blocks, key=lambda b: (b[1], b[0]))
    return [block[4] for block in ordered], 1


def extract_pymupdf(data: bytes) -> ExtractionResult:
    """Extract clinical PDF content using PyMuPDF."""
    start = time.perf_counter()
    pages: list[PageExtraction] = []

    with fitz.open(stream=data, filetype="pdf") as document:
        if document.needs_pass:
            raise ValueError("Encrypted PDF requires a password")

        for number, page in enumerate(document, start=1):
            warnings: list[str] = []
            tables: list[list[list[str]]] = []
            table_boxes: list[tuple[float, float, float, float]] = []

            if hasattr(page, "find_tables"):
                try:
                    detected = page.find_tables()

                    for table in detected.tables:
                        rows = [
                            [_normalize_cell(cell) for cell in row]
                            for row in table.extract()
                        ]

                        if rows:
                            tables.append(rows)
                            table_boxes.append(tuple(table.bbox))

                except (RuntimeError, ValueError, TypeError) as exc:
                    LOGGER.warning(
                        "Table detection failed on page %s: %s",
                        number,
                        exc,
                    )
                    warnings.append("Table detection failed")

            raw_blocks = page.get_text("blocks", sort=False)
            blocks = []

            for block in raw_blocks:
                if len(block) < 5:
                    continue

                if len(block) >= 7 and block[6] != 0:
                    continue

                content = block[4]

                if not isinstance(content, str) or not content.strip():
                    continue

                coordinates = tuple(float(v) for v in block[:4])

                if _intersects(coordinates, table_boxes):
                    continue

                blocks.append((*coordinates, content.strip()))

            parts, columns = _column_groups(
                blocks,
                float(page.rect.width),
            )

            table_sections = [
                "[TABLE]\n" + _render_table(table) + "\n[/TABLE]"
                for table in tables
            ]

            text = "\n\n".join(parts + table_sections).strip()

            if not text:
                warnings.append(
                    "No extractable text; scanned PDF may require OCR"
                )

            pages.append(
                PageExtraction(
                    page=number,
                    text=text,
                    tables=tables,
                    columns_detected=columns,
                    warnings=warnings,
                )
            )

    return ExtractionResult(
        parser="pymupdf",
        pages=pages,
        elapsed_seconds=time.perf_counter() - start,
    )


def extract_pdfplumber(data: bytes) -> ExtractionResult:
    """Extract PDF content using pdfplumber."""
    start = time.perf_counter()
    pages: list[PageExtraction] = []

    with pdfplumber.open(io.BytesIO(data)) as document:
        for number, page in enumerate(document.pages, start=1):
            warnings: list[str] = []
            tables: list[list[list[str]]] = []

            try:
                found_tables = page.find_tables()

                for table in found_tables:
                    rows = [
                        [_normalize_cell(cell) for cell in row]
                        for row in table.extract()
                    ]
                    if rows:
                        tables.append(rows)

                def outside_tables(obj: dict) -> bool:
                    x0 = float(obj.get("x0", 0))
                    x1 = float(obj.get("x1", x0))
                    y0 = float(obj.get("top", 0))
                    y1 = float(obj.get("bottom", y0))

                    cx = (x0 + x1) / 2
                    cy = (y0 + y1) / 2

                    return not any(
                        table.bbox[0] <= cx <= table.bbox[2]
                        and table.bbox[1] <= cy <= table.bbox[3]
                        for table in found_tables
                    )

                prose = page.filter(outside_tables)

                words = prose.extract_words(
                    x_tolerance=2,
                    y_tolerance=3,
                )

                line_groups: dict[int, list[dict]] = {}

                for word in words:
                    key = round(float(word["top"]) / 4)
                    line_groups.setdefault(key, []).append(word)

                blocks = []

                for group in line_groups.values():
                    group.sort(key=lambda w: w["x0"])

                    runs = []
                    current = []

                    for word in group:
                        if current:
                            gap = word["x0"] - current[-1]["x1"]

                            crosses_gutter = (
                                current[-1]["x1"] < page.width * 0.48
                                and word["x0"] > page.width * 0.52
                            )

                            if gap > page.width * 0.075 or crosses_gutter:
                                runs.append(current)
                                current = []

                        current.append(word)

                    if current:
                        runs.append(current)

                    for run in runs:
                        blocks.append(
                            (
                                min(w["x0"] for w in run),
                                min(w["top"] for w in run),
                                max(w["x1"] for w in run),
                                max(w["bottom"] for w in run),
                                " ".join(w["text"] for w in run),
                            )
                        )

                parts, columns = _column_groups(
                    blocks,
                    float(page.width),
                )

                table_sections = [
                    "[TABLE]\n" + _render_table(table) + "\n[/TABLE]"
                    for table in tables
                ]

                text = "\n\n".join(parts + table_sections).strip()

                if not text:
                    warnings.append(
                        "No extractable text; scanned PDF may require OCR"
                    )

                pages.append(
                    PageExtraction(
                        page=number,
                        text=text,
                        tables=tables,
                        columns_detected=columns,
                        warnings=warnings,
                    )
                )

            except (ValueError, RuntimeError, TypeError) as exc:
                LOGGER.exception(
                    "pdfplumber extraction failed on page %s",
                    number,
                )

                pages.append(
                    PageExtraction(
                        page=number,
                        text="",
                        warnings=[
                            f"Page extraction failed: {type(exc).__name__}"
                        ],
                    )
                )

    return ExtractionResult(
        parser="pdfplumber",
        pages=pages,
        elapsed_seconds=time.perf_counter() - start,
    )


def extract_pdf(
    data: bytes,
    parser: ParserName = "pymupdf",
) -> ExtractionResult:
    """Select the requested PDF extraction engine."""
    if not data.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF signature")

    if parser == "pymupdf":
        return extract_pymupdf(data)

    if parser == "pdfplumber":
        return extract_pdfplumber(data)

    raise ValueError(f"Unsupported parser: {parser}")
