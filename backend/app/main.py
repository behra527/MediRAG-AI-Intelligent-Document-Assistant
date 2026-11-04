
"""
MediRAG AI - FastAPI Application.

Clinical guideline PDF parsing, text cleaning,
benchmarking, ChromaDB ingestion, and RAG chat.

Compatible with:
    app.core
    app.llm
    app.rag
    app.services.pdf_extractor
    app.services.text_cleaner
    app.services.compare_parsers

Python 3.13
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import httpx
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query as QueryParam,
    UploadFile,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services.chunking import report, validate
from .core import settings
from .llm import generate
from .rag import (
    delete_document,
    documents,
    ingest_pdf,
    retrieve,
)
from .services.compare_parsers import (
    _metrics,
    compare as compare_pdf_parsers,
)
from .services.pdf_extractor import extract_pdf
from .services.text_cleaner import clean_pages


LOGGER = logging.getLogger(__name__)

ParserName = Literal["pymupdf", "pdfplumber"]
IngestionParser = Literal["auto", "pymupdf", "pdfplumber"]

MAX_FILENAME_LENGTH = 120


# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------

app = FastAPI(
    title="MediRAG AI API",
    description=(
        "Clinical Guideline Intelligence Platform "
        "with PDF parsing, text hygiene, "
        "retrieval and evidence-grounded AI."
    ),
    version="2.0.0",
)


# Local React/Vite development origins are enabled explicitly.
# Additional trusted origins can be supplied using CORS_ORIGINS in .env.
local_dev_origins = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
}
configured_origins = {
    origin.strip().rstrip("/")
    for origin in settings().cors_origins.split(",")
    if origin.strip()
}
allowed_origins = sorted(local_dev_origins | configured_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Request models
# ---------------------------------------------------------

class Query(BaseModel):
    """Clinical question request."""

    question: str = Field(
        min_length=3,
        max_length=3000,
    )

    document_id: str | None = None

    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
    )


# ---------------------------------------------------------
# Upload validation
# ---------------------------------------------------------

async def _read_pdf(
    file: UploadFile,
) -> tuple[bytes, str]:
    """
    Read and validate an uploaded PDF.

    Enforces:
        - .pdf extension
        - upload size limit
        - PDF file signature
        - nonempty file
    """

    filename = (file.filename or "").strip()

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="A PDF filename is required.",
        )

    # Strip client-provided path components.
    filename = filename.replace("\\", "/").split("/")[-1]

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Upload a PDF file.",
        )

    filename = filename[:MAX_FILENAME_LENGTH]

    max_size = settings().max_upload_mb * 1024 * 1024

    if max_size <= 0:
        LOGGER.error("Invalid MAX_UPLOAD_MB configuration")
        raise HTTPException(
            status_code=500,
            detail="Upload service is misconfigured.",
        )

    try:
        data = await file.read(max_size + 1)
    except OSError as exc:
        LOGGER.exception("Failed to read uploaded PDF")
        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded file.",
        ) from exc
    finally:
        await file.close()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="The uploaded PDF is empty.",
        )

    if len(data) > max_size:
        raise HTTPException(
            status_code=413,
            detail=(
                f"PDF exceeds the configured "
                f"{settings().max_upload_mb} MB limit."
            ),
        )

    if not data.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file signature.",
        )

    return data, filename


# ---------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------

@app.get("/api/health")
def health() -> dict:
    """
    Lightweight API health endpoint.

    Does not expose API keys or private configuration.
    """

    return {
        "status": "ok",
        "service": "MediRAG AI",
        "version": app.version,
        "model": settings().openrouter_model,
    }


# ---------------------------------------------------------
# Document management
# ---------------------------------------------------------

@app.get("/api/documents")
async def list_documents() -> list[dict]:
    """List documents indexed in ChromaDB."""

    try:
        return await run_in_threadpool(documents)
    except Exception as exc:
        LOGGER.exception("Document listing failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to list indexed documents.",
        ) from exc


@app.delete("/api/documents/{document_id}")
async def remove(document_id: str) -> dict:
    """Delete a document and its indexed chunks."""

    try:
        deleted = await run_in_threadpool(
            delete_document,
            document_id,
        )
    except Exception as exc:
        LOGGER.exception("Document deletion failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to delete document.",
        ) from exc

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {"deleted": True}


# ---------------------------------------------------------
# PDF upload and ChromaDB ingestion
# ---------------------------------------------------------

@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...),
    strategy: Literal["fixed", "structure"] = "structure",
    chunk_size: int = QueryParam(950, ge=50, le=10000),
    overlap: int = QueryParam(150, ge=0),
    parser: Annotated[
        IngestionParser,
        QueryParam(
            description=(
                "PDF extraction engine. "
                "'auto' prefers PyMuPDF and falls back "
                "to pdfplumber when necessary."
            )
        ),
    ] = "auto",
) -> dict:
    """
    Upload and index a clinical guideline.

    Pipeline:
        PDF validation
        -> extraction
        -> cleaning
        -> chunking
        -> embeddings
        -> ChromaDB
    """

    data, filename = await _read_pdf(file)

    LOGGER.info(
        "Starting PDF ingestion: filename=%s parser=%s",
        filename,
        parser,
    )

    try:
        result = await run_in_threadpool(
            ingest_pdf,
            data,
            filename,
            parser,
            strategy, chunk_size, overlap,
        )

        return result

    except ValueError as exc:
        LOGGER.warning(
            "PDF ingestion rejected: %s",
            exc,
        )
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        LOGGER.exception(
            "Clinical PDF indexing failed"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Document indexing failed. "
                "Check server logs for details."
            ),
        ) from exc


# ---------------------------------------------------------
# PDF parsing and text hygiene
# ---------------------------------------------------------

@app.post("/api/v1/parse-and-clean")
async def parse_and_clean(
    file: UploadFile = File(...),
    parser: Annotated[
        ParserName,
        QueryParam(
            description=(
                "Choose pymupdf or pdfplumber "
                "for PDF extraction."
            )
        ),
    ] = "pymupdf",
    benchmark: Annotated[
        bool,
        QueryParam(
            description=(
                "Run both parsers and return "
                "comparison diagnostics."
            )
        ),
    ] = False,
) -> dict:
    """
    Parse and clean a clinical PDF without indexing.

    Returns:
        - cleaned_text
        - page-level cleaned content
        - structured tables
        - column detection
        - extraction warnings
        - diagnostic metrics
        - optional parser benchmark
    """

    data, filename = await _read_pdf(file)

    LOGGER.info(
        "PDF parsing requested: filename=%s parser=%s",
        filename,
        parser,
    )

    try:
        extracted = await run_in_threadpool(
            extract_pdf,
            data,
            parser,
        )

        cleaned = await run_in_threadpool(
            clean_pages,
            [page.text for page in extracted.pages],
        )

        if not cleaned.text.strip():
            raise HTTPException(
                status_code=422,
                detail=(
                    "No searchable text was found. "
                    "The PDF may require OCR."
                ),
            )

        diagnostics = await run_in_threadpool(
            _metrics,
            extracted,
        )

        benchmark_report = None

        if benchmark:
            benchmark_report = await run_in_threadpool(
                compare_pdf_parsers,
                data,
            )

        pages = []

        for index, page in enumerate(extracted.pages):
            pages.append(
                {
                    "page": page.page,
                    "text": cleaned.pages[index],
                    "tables": page.tables,
                    "columns_detected": (
                        page.columns_detected
                    ),
                    "warnings": page.warnings,
                }
            )

        return {
            "filename": filename,
            "parser": extracted.parser,
            "cleaned_text": cleaned.text,
            "pages": pages,
            "diagnostics": diagnostics,
            "benchmark": benchmark_report,
            "ready_for_chunking": True,
            "indexed": False,
        }

    except HTTPException:
        raise

    except ValueError as exc:
        LOGGER.warning(
            "PDF parsing rejected: %s",
            exc,
        )
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        LOGGER.exception(
            "Clinical PDF parsing failed"
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "PDF parsing failed. "
                "Check server logs for details."
            ),
        ) from exc


# ---------------------------------------------------------
# Chunking experiment (no embeddings/indexing required)
# ---------------------------------------------------------

class ChunkExperiment(BaseModel):
    text: str = Field(min_length=1, max_length=200000)
    strategy: Literal["fixed", "structure"] = "fixed"
    chunk_size: int = Field(default=950, ge=50, le=10000)
    overlap: int = Field(default=150, ge=0)

@app.post("/api/chunking/preview")
def preview_chunking(payload: ChunkExperiment) -> dict:
    try:
        return report(payload.text, payload.strategy, payload.chunk_size, payload.overlap)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post("/api/chunking/compare")
def compare_chunking(payload: ChunkExperiment) -> dict:
    try:
        validate(payload.chunk_size, payload.overlap, payload.strategy)
        configurations = [
            ("fixed", payload.chunk_size, 0),
            ("fixed", payload.chunk_size, payload.overlap),
            ("structure", payload.chunk_size, 0),
            ("structure", payload.chunk_size, payload.overlap),
            ("fixed", max(50, payload.chunk_size // 2), min(payload.overlap, max(50, payload.chunk_size // 2)-1)),
            ("structure", min(10000, payload.chunk_size * 2), payload.overlap),
        ]
        return {"experiments": [report(payload.text, strategy, size, overlap)
                                for strategy,size,overlap in configurations]}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

@app.post("/api/chunking/preview-pdf")
async def preview_pdf_chunks(
    file: UploadFile = File(...),
    strategy: Literal["fixed", "structure"] = "structure",
    chunk_size: int = QueryParam(950, ge=50, le=10000),
    overlap: int = QueryParam(150, ge=0),
    parser: IngestionParser = "auto",
) -> dict:
    try:
        validate(chunk_size, overlap, strategy)
        data, filename = await _read_pdf(file)
        from .rag import _extract_document, _extract_prose
        extracted = await run_in_threadpool(_extract_document, data, parser)
        cleaned = await run_in_threadpool(clean_pages, [p.text for p in extracted.pages])
        pages = []
        for i, page in enumerate(extracted.pages):
            r = report(_extract_prose(cleaned.pages[i]), strategy, chunk_size, overlap)
            pages.append({"page": page.page, **r})
        return {"filename": filename, "pages": pages,
                "total_chunks": sum(p["chunk_count"] for p in pages)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


# ---------------------------------------------------------
# Shared question answering
# ---------------------------------------------------------

async def answer(
    q: Query,
    mode: Literal["rag", "baseline"],
) -> dict:
    """
    Generate either a RAG-grounded or baseline answer.

    RAG:
        Retrieve relevant clinical chunks and send
        them to the configured LLM.

    Baseline:
        Use the LLM without retrieved documents.
    """

    sources = []

    if mode == "rag":
        try:
            sources = await run_in_threadpool(
                retrieve,
                q.question,
                q.top_k,
                q.document_id,
            )
        except Exception as exc:
            LOGGER.exception(
                "Clinical retrieval failed"
            )
            raise HTTPException(
                status_code=500,
                detail="Clinical document retrieval failed.",
            ) from exc

        if not sources:
            return {
                "answer": (
                    "Not found in the uploaded guideline. "
                    "Upload a searchable PDF or "
                    "choose another document."
                ),
                "sources": [],
                "mode": mode,
            }

    try:
        response = await generate(
            q.question,
            sources,
            mode == "rag",
        )

    except httpx.HTTPStatusError as exc:
        LOGGER.warning(
            "LLM provider returned HTTP %s",
            exc.response.status_code,
        )
        raise HTTPException(
            status_code=502,
            detail="LLM provider returned an error.",
        ) from exc

    except (
        httpx.RequestError,
        RuntimeError,
        KeyError,
        IndexError,
        ValueError,
    ) as exc:
        LOGGER.exception(
            "LLM response generation failed"
        )
        raise HTTPException(
            status_code=502,
            detail="Unable to generate an AI response.",
        ) from exc

    return {
        "answer": response,
        "sources": sources,
        "mode": mode,
    }


# ---------------------------------------------------------
# RAG chat
# ---------------------------------------------------------

@app.post("/api/chat")
async def chat(q: Query) -> dict:
    """Answer questions using indexed guidelines."""

    return await answer(q, "rag")


# ---------------------------------------------------------
# RAG vs baseline comparison
# ---------------------------------------------------------

@app.post("/api/compare")
async def compare(q: Query) -> dict:
    """
    Compare baseline and retrieval-grounded answers.

    The comparison is observational.
    It does not automatically establish clinical accuracy.
    """

    baseline = await answer(q, "baseline")
    rag = await answer(q, "rag")

    return {
        "without_rag": baseline,
        "with_rag": rag,
        "note": (
            "Judge correctness against the official "
            "clinical guideline. RAG does not "
            "guarantee factual or clinical accuracy."
        ),
    }
