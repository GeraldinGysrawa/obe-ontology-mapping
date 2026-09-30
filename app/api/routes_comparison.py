"""
API Routes — Skill Comparison: ESCO Skills vs PLO JTK.

Endpoint:
- POST /comparison/esco-skills-vs-plo — Hitung similarity ESCO Skills vs PLO JTK,
  hasilkan daftar covered, gap, dan overskill.
"""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import ComparisonRequest, ComparisonResponse
from app.repositories import jtk_repository
from app.services import skill_comparator, skill_lookup

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/comparison", tags=["Skill Comparison"])


@router.post(
    "/esco-skills-vs-plo",
    response_model=ComparisonResponse,
    summary="Bandingkan ESCO Skills vs PLO JTK",
    description=(
        "Mengambil keterampilan ESCO dari occupation_uri dan daftar PLO JTK "
        "dari peo_id, lalu menghitung SBERT+Cosine Similarity dengan threshold "
        "tertentu untuk menghasilkan analisis keselarasan kurikulum: "
        "Covered (terpenuhi), Gap (kebutuhan industri belum diajarkan), "
        "dan Overskill (materi kurikulum spesifik internal)."
    ),
)
async def compare_esco_skills_vs_plo(request: ComparisonRequest):
    """POST /comparison/esco-skills-vs-plo — Perbandingan ESCO Skills vs PLO JTK."""
    try:
        # 1. Ambil ESCO Skills dari Occupation
        esco_skills = skill_lookup.get_skills_for_occupation(request.occupation_uri)
        if not esco_skills:
            raise HTTPException(
                status_code=404,
                detail=f"Tidak ada ESCO Skills untuk occupation URI '{request.occupation_uri}'.",
            )

        # 2. Ambil PLO dari PEO
        plo_list = jtk_repository.get_plo_list(peo_id=request.peo_id)
        if not plo_list:
            raise HTTPException(
                status_code=404,
                detail=f"Tidak ada PLO untuk PEO ID '{request.peo_id}'.",
            )

        # 3. Hitung similarity + threshold (Covered, Gap, Overskill)
        result = skill_comparator.compare_skills_vs_plo(
            esco_skills=esco_skills,
            plo_list=plo_list,
            threshold=request.threshold,
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error di /comparison/esco-skills-vs-plo: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
