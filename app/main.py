"""
Entrypoint FastAPI — JTK-ESCO Mapping Pipeline.

Pipeline pemetaan kurikulum Jurusan Teknik Komputer (JTK) ke standar
kompetensi kerja Eropa (ESCO) menggunakan SBERT + Cosine Similarity.

Jalankan:
    uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes_clo import router as clo_router
from app.api.routes_comparison import router as comparison_router
from app.api.routes_experiments import router as experiments_router
from app.api.routes_occupation import router as occupation_router
from app.api.routes_pipeline import router as pipeline_router
from app.api.routes_cso import router as cso_router
from app.api.routes_pipeline_cso import router as pipeline_cso_router
from app.core.embeddings import load_model
from app.repositories.cso_repository import CSORepository

# ── Logging setup ──────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Lifespan handler ──────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup & shutdown handler.

    Startup:
    - Load model SBERT SEKALI (singleton) supaya tidak di-load ulang
      di setiap request. Ini juga memastikan perbandingan waktu eksekusi
      antar metode eksperimen adil (tidak bias karena overhead loading).
    """
    logger.info("=" * 60)
    logger.info("Starting JTK-ESCO Mapping Pipeline ...")
    logger.info("=" * 60)

    # Load SBERT model saat startup
    load_model()
    
    # Load CSO Repository
    logger.info("Memuat CSO Repository... (ini mungkin membutuhkan beberapa saat)")
    CSORepository().load_data()

    logger.info("Startup selesai. Pipeline siap menerima request.")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down JTK-ESCO Mapping Pipeline.")


# ── FastAPI app ────────────────────────────────────────────────────────────

app = FastAPI(
    title="JTK-ESCO Mapping Pipeline",
    description=(
        "Backend pipeline pemetaan kurikulum Jurusan Teknik Komputer (JTK) "
        "ke standar kompetensi kerja Eropa (ESCO). "
        "Menggunakan SBERT (paraphrase-multilingual-MiniLM-L12-v2) + "
        "Cosine Similarity untuk semantic matching."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── Include routers ───────────────────────────────────────────────────────

app.include_router(occupation_router)
app.include_router(comparison_router)
app.include_router(clo_router)
app.include_router(experiments_router)
app.include_router(pipeline_router)
app.include_router(cso_router)
app.include_router(pipeline_cso_router)


# ── Health check ───────────────────────────────────────────────────────────


@app.get(
    "/health",
    tags=["System"],
    summary="Health check",
)
async def health_check():
    """GET /health — Health check endpoint."""
    return {"status": "ok", "service": "jtk-esco-mapping"}
