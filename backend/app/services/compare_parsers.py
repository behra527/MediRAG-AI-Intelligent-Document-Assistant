
"""
MediRAG AI - Clinical PDF Parser Benchmarking.

Compare PyMuPDF and pdfplumber on the same PDF.

Metrics are reference-free diagnostics, not validated
clinical extraction accuracy.

Compatible with:
    app.services.pdf_extractor
    app.services.text_cleaner
    app.main

Python 3.13
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import statistics
import time

from pathlib import Path
from typing import Any

from .pdf_extractor import ExtractionResult, extract_pdf
from .text_cleaner import CleanResult, clean_pages


LOGGER = logging.getLogger(__name__)

PARSERS = ("pymupdf", "pdfplumber")

WORD_PATTERN = re.compile(r"\b[\w]+(?:[-'][\w]+)*\b")
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

CLINICAL_UNIT_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?\s*"
    r"(?:mg|mcg|µg|μg|g|kg|mL|ml|mmol|IU|units?)\b",
    re.IGNORECASE,
)

WARNING_PATTERN = re.compile(
    r"\b(?:warning|caution|contraindication|"
    r"do not|avoid|urgent|emergency)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------
# Text yield metrics
# ---------------------------------------------------------

def _text_metrics(text: str) -> dict[str, Any]:
    """
    Count text yield using deterministic regex rules.

    Token estimates are not GPT/OpenRouter tokenizer
    counts and must not be used for exact context limits.
    """

    words = WORD_PATTERN.findall(text)
    tokens = TOKEN_PATTERN.findall(text)

    return {
        "characters": len(text),
        "characters_without_spaces": len(
            re.sub(r"\s", "", text)
        ),
        "words": len(words),
        "tokens_estimated": len(tokens),
        "paragraphs": len(
            [
                item
                for item in re.split(r"\n\s*\n", text)
                if item.strip()
            ]
        ),
        "clinical_unit_mentions": len(
            CLINICAL_UNIT_PATTERN.findall(text)
        ),
        "warning_keyword_mentions": len(
            WARNING_PATTERN.findall(text)
        ),
    }


# ---------------------------------------------------------
# Table diagnostics
# ---------------------------------------------------------

def _table_metrics(
    result: ExtractionResult,
) -> dict[str, Any]:
    """
    Evaluate detected table structure.

    These metrics cannot establish whether a medication
    dose or table header was extracted correctly.
    """

    tables = [
        table
        for page in result.pages
        for table in page.tables
    ]

    row_counts: list[int] = []
    column_counts: list[int] = []

    total_cells = 0
    populated_cells = 0
    empty_cells = 0

    irregular_tables = 0
    total_rows = 0

    for table in tables:
        row_counts.append(len(table))
        total_rows += len(table)

        widths = [len(row) for row in table]

        if widths:
            column_counts.append(max(widths))

            if len(set(widths)) > 1:
                irregular_tables += 1

        for row in table:
            for cell in row:
                total_cells += 1

                if str(cell or "").strip():
                    populated_cells += 1
                else:
                    empty_cells += 1

    pages_with_tables = sum(
        bool(page.tables)
        for page in result.pages
    )

    return {
        "tables_detected": len(tables),
        "pages_with_tables": pages_with_tables,
        "table_rows": total_rows,
        "table_cells": total_cells,
        "populated_cells": populated_cells,
        "empty_cells": empty_cells,
        "populated_cell_ratio": (
            round(populated_cells / total_cells, 3)
            if total_cells
            else None
        ),
        "empty_cell_ratio": (
            round(empty_cells / total_cells, 3)
            if total_cells
            else None
        ),
        "irregular_tables": irregular_tables,
        "average_rows_per_table": (
            round(statistics.mean(row_counts), 2)
            if row_counts
            else 0
        ),
        "average_columns_per_table": (
            round(statistics.mean(column_counts), 2)
            if column_counts
            else 0
        ),
        "structure_preservation_note": (
            "Detected rows, cells and consistency are "
            "structural proxies, not ground-truth accuracy."
        ),
    }


# ---------------------------------------------------------
# Multi-column diagnostics
# ---------------------------------------------------------

def _layout_metrics(
    result: ExtractionResult,
) -> dict[str, Any]:
    """
    Summarize geometric column detection.

    Column counts do not independently prove that
    sentences were reconstructed in the correct order.
    """

    total_pages = len(result.pages)

    two_column_pages = [
        page.page
        for page in result.pages
        if page.columns_detected == 2
    ]

    single_column_pages = [
        page.page
        for page in result.pages
        if page.columns_detected == 1
    ]

    empty_pages = [
        page.page
        for page in result.pages
        if not page.text.strip()
    ]

    pages_with_warnings = [
        page.page
        for page in result.pages
        if page.warnings
    ]

    return {
        "pages_with_two_columns": len(two_column_pages),
        "two_column_page_numbers": two_column_pages,
        "pages_with_one_column": len(single_column_pages),
        "single_column_page_numbers": single_column_pages,
        "empty_pages": empty_pages,
        "pages_with_warnings": pages_with_warnings,
        "two_column_detection_ratio": (
            round(len(two_column_pages) / total_pages, 3)
            if total_pages
            else 0.0
        ),
        "reading_order_verified": False,
        "readability_note": (
            "Reading-order correctness requires manual "
            "inspection or annotated reference text."
        ),
    }


# ---------------------------------------------------------
# Noise and cleaning diagnostics
# ---------------------------------------------------------

def _cleaning_metrics(
    cleaned: CleanResult,
) -> dict[str, Any]:
    """
    Return text hygiene diagnostics.

    A larger removed-line count is not automatically
    better because clinically meaningful text could
    otherwise be removed.
    """

    return {
        "headers_footers_removed": cleaned.removed_lines,
        "removed_examples": cleaned.removed_examples,
        "hyphen_repairs": cleaned.hyphen_repairs,
        "cleaning_warnings": cleaned.warnings,
    }


# ---------------------------------------------------------
# Main metrics
# ---------------------------------------------------------

def _metrics(
    result: ExtractionResult,
) -> dict[str, Any]:
    """
    Backward-compatible metrics function.

    main.py currently imports _metrics directly.
    Existing flat metric keys are retained.
    """

    cleaning_start = time.perf_counter()

    cleaned = clean_pages(
        [page.text for page in result.pages]
    )

    cleaning_seconds = (
        time.perf_counter() - cleaning_start
    )

    text = cleaned.text

    yield_metrics = _text_metrics(text)
    table_metrics = _table_metrics(result)
    layout_metrics = _layout_metrics(result)
    cleaning_metrics = _cleaning_metrics(cleaned)

    extraction_seconds = result.elapsed_seconds

    total_seconds = (
        extraction_seconds + cleaning_seconds
    )

    page_count = len(result.pages)

    metrics: dict[str, Any] = {
        "parser": result.parser,
        "elapsed_seconds": round(
            extraction_seconds,
            4,
        ),
        "cleaning_seconds": round(
            cleaning_seconds,
            4,
        ),
        "total_processing_seconds": round(
            total_seconds,
            4,
        ),
        "pages": page_count,
        "pages_per_second": (
            round(page_count / total_seconds, 2)
            if total_seconds > 0
            else None
        ),

        # Existing compatibility fields.
        "tables_detected": table_metrics["tables_detected"],
        "table_rows": table_metrics["table_rows"],
        "table_cells": table_metrics["table_cells"],
        "populated_cell_ratio": table_metrics[
            "populated_cell_ratio"
        ],
        "headers_footers_removed": cleaned.removed_lines,
        "removed_examples": cleaned.removed_examples,
        "hyphen_repairs": cleaned.hyphen_repairs,
        "pages_with_two_columns": layout_metrics[
            "pages_with_two_columns"
        ],
        "characters": yield_metrics["characters"],
        "tokens_estimated": yield_metrics[
            "tokens_estimated"
        ],
        "empty_pages": layout_metrics["empty_pages"],
        "warnings": {
            page.page: page.warnings
            for page in result.pages
            if page.warnings
        },

        # Detailed diagnostics.
        "performance": {
            "extraction_seconds": round(
                extraction_seconds, 4
            ),
            "cleaning_seconds": round(
                cleaning_seconds, 4
            ),
            "total_seconds": round(
                total_seconds, 4
            ),
        },
        "table_analysis": table_metrics,
        "noise_analysis": cleaning_metrics,
        "layout_analysis": layout_metrics,
        "text_yield": yield_metrics,
    }

    return metrics


# ---------------------------------------------------------
# Heuristic parser recommendation
# ---------------------------------------------------------

def _recommend_parser(
    metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """
    Recommend a candidate parser for further validation.

    This is not a clinical quality score.

    More detected tables or more extracted characters
    do not necessarily indicate higher accuracy.
    """

    available = {
        name: data
        for name, data in metrics.items()
        if data.get("status") == "success"
    }

    if not available:
        return {
            "prose_default": None,
            "table_candidate": None,
            "overall_candidate": None,
            "reason": "Both parsers failed.",
            "requires_manual_validation": True,
        }

    if len(available) == 1:
        name = next(iter(available))

        return {
            "prose_default": name,
            "table_candidate": name,
            "overall_candidate": name,
            "reason": (
                "Only one parser completed successfully."
            ),
            "requires_manual_validation": True,
        }

    pymupdf = available["pymupdf"]
    plumber = available["pdfplumber"]

    pymupdf_tables = pymupdf["tables_detected"]
    plumber_tables = plumber["tables_detected"]

    if plumber_tables > pymupdf_tables:
        table_candidate = "pdfplumber"
    elif pymupdf_tables > plumber_tables:
        table_candidate = "pymupdf"
    else:
        table_candidate = None

    prose_default = "pymupdf"

    if (
        pymupdf["characters"] == 0
        and plumber["characters"] > 0
    ):
        prose_default = "pdfplumber"

    overall_candidate = prose_default

    if (
        pymupdf["characters"] == 0
        and plumber["characters"] == 0
    ):
        overall_candidate = None

    return {
        "prose_default": prose_default,
        "table_candidate": table_candidate,
        "overall_candidate": overall_candidate,
        "reason": (
            "PyMuPDF is the default prose candidate. "
            "Table candidate is based on detected table "
            "counts only, not extraction accuracy."
        ),
        "requires_manual_validation": True,
    }


# ---------------------------------------------------------
# Benchmark both parsers
# ---------------------------------------------------------

def compare(data: bytes) -> dict[str, Any]:
    """
    Run both extraction engines on the same PDF.

    Each parser is isolated so one failure does not
    prevent reporting the other parser's diagnostics.
    """

    if not isinstance(data, bytes):
        raise TypeError(
            "PDF data must be bytes"
        )

    if not data.startswith(b"%PDF-"):
        raise ValueError(
            "Invalid PDF signature"
        )

    results: dict[str, dict[str, Any]] = {}

    for parser_name in PARSERS:
        LOGGER.info(
            "Starting clinical PDF benchmark: %s",
            parser_name,
        )

        try:
            extracted = extract_pdf(
                data,
                parser_name,
            )

            parser_metrics = _metrics(extracted)

            parser_metrics["status"] = "success"

            results[parser_name] = parser_metrics

            LOGGER.info(
                "Benchmark completed: %s",
                parser_name,
            )

        except Exception as exc:
            LOGGER.exception(
                "Benchmark failed for parser %s",
                parser_name,
            )

            results[parser_name] = {
                "parser": parser_name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc)[:250],
            }

    successful = [
        name
        for name, result in results.items()
        if result["status"] == "success"
    ]

    recommendation = _recommend_parser(results)

    return {
        "benchmark_name": (
            "MediRAG AI Clinical PDF Parser Comparison"
        ),
        "benchmark_version": "2.0",
        "pages_evaluated": (
            results[successful[0]]["pages"]
            if successful
            else 0
        ),
        "metrics": results,
        "interpretation": {
            **recommendation,
            "reference_accuracy_available": False,
            "note": (
                "This benchmark measures extraction "
                "performance and structural proxies. "
                "It does not measure clinical accuracy. "
                "Medication doses, units, warnings, "
                "tables and reading order must be "
                "validated against reference pages."
            ),
        },
    }


# ---------------------------------------------------------
# JSON report generation
# ---------------------------------------------------------

def save_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """
    Save a benchmark report as formatted JSON.
    """

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return destination


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main() -> None:
    """
    Run from project backend directory:

    python -m app.services.compare_parsers guideline.pdf

    Or save a JSON report:

    python -m app.services.compare_parsers guideline.pdf \
        --output benchmark.json
    """

    parser = argparse.ArgumentParser(
        description=(
            "Benchmark PyMuPDF and pdfplumber "
            "for MediRAG clinical PDF extraction."
        )
    )

    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to clinical PDF",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path",
    )

    args = parser.parse_args()

    if not args.pdf.is_file():
        parser.error(
            f"PDF file not found: {args.pdf}"
        )

    data = args.pdf.read_bytes()

    report = compare(data)

    if args.output:
        destination = save_report(
            report,
            args.output,
        )

        print(
            f"Benchmark report saved to: {destination}"
        )
    else:
        print(
            json.dumps(
                report,
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
