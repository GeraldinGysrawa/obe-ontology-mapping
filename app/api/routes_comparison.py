"""
API Routes — Comparison (Step D↔E → F/G).

Endpoint:
- POST /comparison/plo-vs-esco — Hitung similarity ESCO Skills vs PLO JTK,
  hasilkan daftar covered (F) dan gap (G).
"""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import ComparisonRequest, ComparisonResponse
from app.repositories import jtk_repository
from app.services import skill_comparator, skill_lookup

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comparison", tags=["Comparison (D↔E → F/G)"])


@router.post(
    "/plo-vs-esco",
    response_model=ComparisonResponse,
    summary="Step D↔E → F/G: Bandingkan ESCO Skills vs PLO JTK",
    description=(
        "Ambil ESCO Skills dari occupation (D), ambil PLO dari PEO (E), "
        "hitung SBERT+Cosine similarity, terapkan threshold untuk "
        "menghasilkan daftar covered (F) dan gap (G)."
    ),
)
async def compare_plo_vs_esco(request: ComparisonRequest):
    """POST /comparison/plo-vs-esco — Step D↔E → F/G."""
    try:
        # Step C→D: Ambil ESCO Skills
        esco_skills = skill_lookup.get_skills_for_occupation(request.occupation_uri)
        if not esco_skills:
            raise HTTPException(
                status_code=404,
                detail=f"Tidak ada ESCO Skills untuk occupation URI '{request.occupation_uri}'.",
            )

        # Step A→E: Ambil PLO dari PEO
        plo_list = jtk_repository.get_plo_list(peo_id=request.peo_id)
        if not plo_list:
            raise HTTPException(
                status_code=404,
                detail=f"Tidak ada PLO untuk PEO ID '{request.peo_id}'.",
            )

        # Step D↔E → F/G: Hitung similarity + threshold
        result = skill_comparator.compare_skills_vs_plo(
            esco_skills=esco_skills,
            plo_list=plo_list,
            threshold=request.threshold,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error di /comparison/plo-vs-esco: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
