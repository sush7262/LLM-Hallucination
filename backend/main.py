"""
FastAPI application — Claim-Level Hallucination Verification Module

Endpoints:
  POST /verify  — Run the 4-stage verification pipeline
  GET  /health  — Liveness check with device info

Models are loaded ONCE at startup via the lifespan context manager
and stay in memory/GPU for all subsequent requests.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.models import VerifyRequest, VerifyResponse, HealthResponse
from backend.pipeline.aligner import RetrievalAligner
from backend.pipeline.classifier import EntailmentClassifier
from backend.pipeline.verifier import HallucinationVerifier
from backend.pdf_parser import extract_passages_from_pdf

# ── Logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-30s │ %(levelname)-7s │ %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan: load models once at startup ───────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load heavy ML models before the first request; clean up on shutdown."""
    device = settings.resolved_device
    logger.info(f"═══ Starting up — device: {device} ═══")

    # Stage 2 model — SentenceTransformer for retrieval alignment
    aligner = RetrievalAligner(
        model_name=settings.embedding_model,
        device=device,
    )

    # Stage 3 model — DeBERTa for entailment classification
    classifier = EntailmentClassifier(
        model_name=settings.nli_model,
        device=device,
        max_length=settings.nli_max_length,
    )

    # Orchestrator
    verifier = HallucinationVerifier(aligner=aligner, classifier=classifier)

    # Store in app.state so endpoints can access them
    app.state.verifier = verifier
    app.state.device = device

    logger.info("═══ All models loaded — ready to serve ═══")
    yield
    logger.info("═══ Shutting down ═══")


# ── App ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Hallucination Verification Module",
    description=(
        "Claim-level hallucination detection for RAG systems. "
        "Takes a generated answer + retrieved passages and returns "
        "per-claim verdicts (supported / unsupported / contradicted)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend to call from any origin (dev convenience)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ───────────────────────────────────────────────────────────

@app.post("/verify", response_model=VerifyResponse)
async def verify(request: VerifyRequest):
    """
    Run the 4-stage hallucination verification pipeline.

    Accepts a RAG-generated answer and its retrieved passages,
    returns per-claim verdicts and an overall decision.
    """
    try:
        verifier: HallucinationVerifier = app.state.verifier
        result = verifier.verify(
            answer=request.answer,
            passages=request.passages,
            decomposition_mode=request.decomposition_mode,
            config=settings,
        )
        return result
    except Exception as e:
        logger.exception("Verification pipeline error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-pdf")
async def extract_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF document to extract passages page by page.
    Returns the extracted list of passages and document metadata.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        contents = await file.read()
        passages, total_pages = extract_passages_from_pdf(contents)

        if not passages:
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from the uploaded PDF."
            )

        return {
            "filename": file.filename,
            "total_pages": total_pages,
            "total_passages": len(passages),
            "passages": passages,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("PDF extraction error")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness check — returns status and compute device."""
    return HealthResponse(
        status="ok",
        device=app.state.device,
    )


# ── Static Files (Frontend UI) ──────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


