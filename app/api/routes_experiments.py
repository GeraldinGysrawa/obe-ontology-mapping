"""
API Routes — Experiments (perbandingan 3 metode E→K).

Endpoint:
- POST /experiments/compare/{plo_id} — Jalankan 3 metode untuk satu PLO
- POST /experiments/export — Export hasil ke CSV untuk evaluasi manual
"""

import logging

from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.experiments.export_for_review import export_to_csv
from app.experiments.runner import run_all_methods
from app.experiments.schemas import ExperimentExportRequest, ExperimentRunResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/experiments", tags=["Experiments (3 Metode E→K)"])


@router.post(
    "/compare/{plo_id}",
    response_model=ExperimentRunResult,
    summary="Jalankan 3 metode eksperimen untuk satu PLO",
    description=(
        "Jalankan ketiga metode (SBERT saja, NER saja, NER+SBERT) "
        "untuk satu PLO tertentu. Return hasil JSON lengkap untuk "
        "cek cepat via Swagger UI."
    ),
)
async def compare_methods(plo_id: str):
    """POST /experiments/compare/{plo_id}."""
    try:
        result = await run_all_methods(plo_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Error di /experiments/compare/%s: %s", plo_id, str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/export",
    summary="Export hasil eksperimen ke CSV",
    description=(
        "Jalankan ketiga metode untuk semua PLO yang diberikan, "
        "ekspor hasilnya ke CSV di data/processed/experiment_review.csv. "
        "Kosongkan plo_ids untuk menjalankan semua PLO."
    ),
)
async def export_experiments(request: ExperimentExportRequest):
    """POST /experiments/export."""
    try:
        settings = get_settings()
        output_path = f"{settings.embeddings_cache_dir}/experiment_review.csv"

        result_path = await export_to_csv(
            plo_ids=request.plo_ids,
            output_path=output_path,
        )

        return {
            "status": "success",
            "output_path": result_path,
            "message": (
                f"Hasil eksperimen berhasil diekspor ke {result_path}. "
                "Buka file CSV di Excel, isi kolom 'relevan_manual' dan 'catatan' "
                "untuk evaluasi manual."
            ),
        }
    except Exception as e:
        logger.error("Error di /experiments/export: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
