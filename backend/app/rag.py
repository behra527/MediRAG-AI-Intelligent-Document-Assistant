
"""
MediRAG AI - Clinical Retrieval-Augmented Generation.

Production-oriented PDF ingestion, cleaning, chunking,
embedding, indexing and semantic retrieval.

Compatible with existing:
    app.main
    app.llm
    app.core
    app.services.pdf_extractor
    app.services.text_cleaner

Python 3.13
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from functools import lru_cache
from typing import Any, Literal

import chromadb
from sentence_transformers import SentenceTransformer

from .services.chunking import chunk_text, validate
from .core import STORAGE
from .services.pdf_extractor import extract_pdf
from .services.text_cleaner import clean_pages


LOGGER = logging.getLogger(__name__)

EMBEDDING_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

COLLECTION_NAME = "guidelines"

CHUNK_SIZE = 950
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 64

ParserMode = Literal[
    "auto",
    "pymupdf",
    "pdfplumber",
]

# Protect ingestion and deletion from concurrent writes
# inside this application process.
_WRITE_LOCK = threading.RLock()


# ---------------------------------------------------------
# Embedding model
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def embedder() -> SentenceTransformer:
    """
    Load the embedding model once per Python process.

    The model is reused for indexing and retrieval,
    ensuring consistent vector dimensions.
    """

    LOGGER.info(
        "Loading embedding model: %s",
        EMBEDDING_MODEL,
    )

    return SentenceTransformer(EMBEDDING_MODEL)


# ---------------------------------------------------------
# ChromaDB collection
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def collection():
    """
    Initialize persistent ChromaDB storage.

    Uses cosine distance for normalized embeddings.
    """

    storage_path = STORAGE / "chroma"

    storage_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = chromadb.PersistentClient(
        path=str(storage_path)
    )

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
        },
    )


# ---------------------------------------------------------
# Text chunking
# ---------------------------------------------------------

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[str]:
    """
    Split clinical text into overlapping chunks.

    Preference:
        1. Paragraph boundary
        2. Sentence boundary
        3. Word boundary

    Character-based chunking is retained for
    compatibility with the existing application.
    """

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be positive"
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Invalid chunk overlap"
        )

    if not isinstance(text, str):
        raise TypeError(
            "text must be a string"
        )

    text = text.strip()

    if not text:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(
            start + chunk_size,
            len(text),
        )

        if end < len(text):
            minimum = start + int(
                chunk_size * 0.60
            )

            # Prefer paragraph boundary.
            boundary = text.rfind(
                "\n\n",
                minimum,
                end,
            )

            if boundary > start:
                end = boundary

            else:
                # Prefer sentence-ending punctuation.
                candidates = [
                    text.rfind(
                        marker,
                        minimum,
                        end,
                    )
                    for marker in (
                        ". ",
                        "? ",
                        "! ",
                        ".\n",
                    )
                ]

                boundary = max(candidates)

                if boundary > start:
                    end = boundary + 1

                else:
                    boundary = text.rfind(
                        " ",
                        minimum,
                        end,
                    )

                    if boundary > start:
                        end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(
            start + 1,
            end - overlap,
        )

        # Avoid beginning the next chunk with whitespace.
        while (
            next_start < len(text)
            and text[next_start].isspace()
        ):
            next_start += 1

        start = next_start

    return chunks


# ---------------------------------------------------------
# Table-aware chunking
# ---------------------------------------------------------

def _table_to_text(
    table: list[list[str]],
) -> str:
    """
    Convert a structured clinical table to text.

    Each row retains its cell boundaries.
    """

    rows = []

    for row in table:
        normalized = [
            " ".join(str(cell or "").split())
            for cell in row
        ]

        rows.append(
            " | ".join(normalized)
        )

    return "\n".join(rows)


def _table_chunks(
    table: list[list[str]],
    chunk_size: int = CHUNK_SIZE,
) -> list[str]:
    """
    Create table-aware chunks.

    The first row is repeated as a header when a
    table is split across multiple chunks.

    Assumption: first row is a header candidate.
    It must be verified for headerless clinical tables.
    """

    if not table:
        return []

    rows = [
        [
            " ".join(str(cell or "").split())
            for cell in row
        ]
        for row in table
    ]

    if len(rows) == 1:
        return [
            "[TABLE]\n"
            + _table_to_text(rows)
            + "\n[/TABLE]"
        ]

    header = rows[0]
    body = rows[1:]

    header_text = " | ".join(header)

    chunks = []
    current_rows = []

    def render(rows_to_render: list[list[str]]) -> str:
        return (
            "[TABLE]\n"
            + header_text
            + "\n"
            + _table_to_text(rows_to_render)
            + "\n[/TABLE]"
        )

    for row in body:
        candidate = current_rows + [row]

        if (
            current_rows
            and len(render(candidate)) > chunk_size
        ):
            chunks.append(render(current_rows))
            current_rows = [row]
        else:
            current_rows = candidate

    if current_rows:
        chunks.append(render(current_rows))

    return chunks


# ---------------------------------------------------------
# Remove duplicated table text from prose
# ---------------------------------------------------------

_TABLE_BLOCK_RE = re.compile(
    r"\[TABLE\].*?\[/TABLE\]",
    flags=re.DOTALL,
)


def _extract_prose(text: str) -> str:
    """
    Remove structured table blocks from prose.

    Tables are indexed separately to avoid
    duplicate embeddings.
    """

    return _TABLE_BLOCK_RE.sub(
        "\n\n",
        text,
    ).strip()


# ---------------------------------------------------------
# Parser selection
# ---------------------------------------------------------

def _extract_document(
    data: bytes,
    parser: ParserMode,
):
    """
    Extract a PDF using the requested parser.

    Auto mode prefers PyMuPDF, but falls back to
    pdfplumber if PyMuPDF returns no searchable text.

    It does not claim to select the clinically
    most accurate parser.
    """

    if parser != "auto":
        return extract_pdf(data, parser)

    try:
        primary = extract_pdf(
            data,
            "pymupdf",
        )

        if any(
            page.text.strip()
            for page in primary.pages
        ):
            return primary

    except (ValueError, RuntimeError) as exc:
        LOGGER.warning(
            "PyMuPDF extraction failed: %s",
            exc,
        )

    LOGGER.info(
        "Falling back to pdfplumber"
    )

    return extract_pdf(
        data,
        "pdfplumber",
    )


# ---------------------------------------------------------
# Prepare document chunks
# ---------------------------------------------------------

def _prepare_chunks(
    extracted,
    filename: str,
    document_id: str,
    strategy: str = "structure",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Convert extracted PDF pages into indexed records.

    Every chunk retains:
        document_id
        filename
        page
        chunk
        content_type
        parser
    """

    cleaned = clean_pages(
        [page.text for page in extracted.pages]
    )

    records: list[dict[str, Any]] = []

    for index, page in enumerate(extracted.pages):
        page_text = cleaned.pages[index]

        prose = _extract_prose(page_text)

        page_chunks = chunk_text(prose, strategy, chunk_size, overlap)

        chunk_index = 0

        for chunk in page_chunks:
            records.append(
                {
                    "id": (
                        f"{document_id}:"
                        f"{page.page}:"
                        f"{chunk_index}"
                    ),
                    "text": chunk.text,
                    "metadata": {
                        "document_id": document_id,
                        "filename": filename,
                        "page": page.page,
                        "chunk": chunk_index,
                        "content_type": "prose",
                        "chunk_strategy": strategy,
                        "chunk_size": chunk_size,
                        "chunk_overlap": overlap,
                        "start_char": chunk.start,
                        "end_char": chunk.end,
                        "parser": extracted.parser,
                    },
                }
            )

            chunk_index += 1

        # Structured tables are indexed independently.
        for table_index, table in enumerate(
            page.tables
        ):
            for table_chunk in _table_chunks(table):
                records.append(
                    {
                        "id": (
                            f"{document_id}:"
                            f"{page.page}:"
                            f"{chunk_index}"
                        ),
                        "text": table_chunk,
                        "metadata": {
                            "document_id": document_id,
                            "filename": filename,
                            "page": page.page,
                            "chunk": chunk_index,
                            "content_type": "table",
                            "table_index": table_index,
                            "parser": extracted.parser,
                        },
                    }
                )

                chunk_index += 1

    return records


# ---------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------

def ingest_pdf(
    data: bytes,
    filename: str,
    parser: ParserMode = "auto",
    strategy: str = "structure",
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> dict[str, Any]:
    """
    Ingest a clinical PDF into ChromaDB.

    Workflow:
        PDF bytes
          -> extraction
          -> clinical text cleaning
          -> prose/table-aware chunking
          -> SentenceTransformer embeddings
          -> persistent ChromaDB storage

    Existing main.py remains compatible:
        ingest_pdf(data, filename)
    """

    if not isinstance(data, bytes):
        raise TypeError(
            "PDF data must be bytes"
        )

    if not data.startswith(b"%PDF-"):
        raise ValueError(
            "Invalid PDF signature"
        )

    if not filename:
        raise ValueError(
            "Filename is required"
        )

    if parser not in (
        "auto",
        "pymupdf",
        "pdfplumber",
    ):
        raise ValueError(
            f"Unsupported parser: {parser}"
        )

    validate(chunk_size, overlap, strategy)

    document_id = hashlib.sha256(
        data
    ).hexdigest()[:20]

    LOGGER.info(
        "Starting clinical PDF ingestion: %s",
        filename,
    )

    try:
        extracted = _extract_document(
            data,
            parser,
        )
    except Exception as exc:
        LOGGER.exception(
            "PDF extraction failed"
        )
        raise ValueError(
            "Cannot extract PDF content"
        ) from exc

    records = _prepare_chunks(
        extracted,
        filename,
        document_id,
        strategy, chunk_size, overlap,
    )

    if not records:
        raise ValueError(
            "No extractable text. "
            "Scanned PDFs may require OCR."
        )

    # Generate embeddings before deleting old records.
    # This avoids losing an existing document when
    # model inference fails.
    prepared_batches = []

    for offset in range(
        0,
        len(records),
        EMBEDDING_BATCH_SIZE,
    ):
        batch = records[
            offset:offset + EMBEDDING_BATCH_SIZE
        ]

        texts = [
            item["text"]
            for item in batch
        ]

        vectors = embedder().encode(
            texts,
            normalize_embeddings=True,
        ).tolist()

        prepared_batches.append(
            (batch, vectors)
        )

    col = collection()

    with _WRITE_LOCK:
        # Replace existing records with the same hash.
        old_ids = col.get(
            where={
                "document_id": document_id
            }
        )["ids"]

        # Reuse stable IDs for identical PDF content.
        for batch, vectors in prepared_batches:
            col.upsert(
                ids=[
                    item["id"]
                    for item in batch
                ],
                documents=[
                    item["text"]
                    for item in batch
                ],
                metadatas=[
                    item["metadata"]
                    for item in batch
                ],
                embeddings=vectors,
            )

        new_ids = {
            item["id"]
            for item in records
        }

        stale_ids = [
            old_id
            for old_id in old_ids
            if old_id not in new_ids
        ]

        if stale_ids:
            col.delete(ids=stale_ids)

    LOGGER.info(
        "Clinical PDF indexed: %s, pages=%s, chunks=%s",
        filename,
        len(extracted.pages),
        len(records),
    )

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": len(extracted.pages),
        "chunks": len(records),
        "parser": extracted.parser,
        "chunk_strategy": strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": overlap,
        "tables_detected": sum(
            len(page.tables)
            for page in extracted.pages
        ),
        "warnings": {
            page.page: page.warnings
            for page in extracted.pages
            if page.warnings
        },
        "status": "indexed",
    }


# ---------------------------------------------------------
# Semantic retrieval
# ---------------------------------------------------------

def retrieve(
    query: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant guideline chunks.

    Preserves the response fields expected by llm.py:
        id
        text
        filename
        page
        score
    """

    if not query.strip():
        return []

    if top_k < 1:
        raise ValueError(
            "top_k must be at least 1"
        )

    col = collection()

    if col.count() == 0:
        return []

    if document_id:
        matching = col.get(
            where={
                "document_id": document_id
            }
        )

        available = len(matching["ids"])

        if available == 0:
            return []
    else:
        available = col.count()

    vector = embedder().encode(
        [query],
        normalize_embeddings=True,
    ).tolist()[0]

    options: dict[str, Any] = {
        "query_embeddings": [vector],
        "n_results": min(
            top_k,
            available,
        ),
        "include": [
            "documents",
            "metadatas",
            "distances",
        ],
    }

    if document_id:
        options["where"] = {
            "document_id": document_id
        }

    try:
        result = col.query(**options)
    except Exception as exc:
        LOGGER.exception(
            "ChromaDB retrieval failed"
        )
        raise RuntimeError(
            "Clinical document retrieval failed"
        ) from exc

    if not result.get("ids"):
        return []

    sources = []

    for item_id, text, metadata, distance in zip(
        result["ids"][0],
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        sources.append(
            {
                "id": item_id,
                "text": text,
                "filename": metadata.get(
                    "filename",
                    "Unknown",
                ),
                "page": metadata.get(
                    "page",
                    0,
                ),
                "score": round(
                    1.0 - float(distance),
                    3,
                ),
                "content_type": metadata.get(
                    "content_type",
                    "prose",
                ),
                "parser": metadata.get(
                    "parser",
                    "unknown",
                ),
            }
        )

    return sources


# ---------------------------------------------------------
# List indexed documents
# ---------------------------------------------------------

def documents() -> list[dict[str, Any]]:
    """
    List indexed clinical documents.

    Compatible with GET /api/documents.
    """

    col = collection()

    result = col.get(
        include=["metadatas"]
    )

    found: dict[str, dict[str, Any]] = {}

    for metadata in result["metadatas"]:
        document_id = metadata["document_id"]

        if document_id not in found:
            found[document_id] = {
                "document_id": document_id,
                "filename": metadata["filename"],
                "chunks": 0,
            }

        found[document_id]["chunks"] += 1

    return list(found.values())


# ---------------------------------------------------------
# Delete indexed document
# ---------------------------------------------------------

def delete_document(
    doc_id: str,
) -> bool:
    """
    Delete all chunks belonging to one document.

    Compatible with DELETE /api/documents/{document_id}.
    """

    col = collection()

    with _WRITE_LOCK:
        result = col.get(
            where={
                "document_id": doc_id
            }
        )

        ids = result["ids"]

        if not ids:
            return False

        col.delete(ids=ids)

    LOGGER.info(
        "Deleted clinical document: %s",
        doc_id,
    )

    return True
