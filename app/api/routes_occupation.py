"""
API Routes — Occupation Matcher.

Endpoint:
- POST /occupation/match — PEO JTK → top-K ESCO Occupation (SBERT+Cosine)
"""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    OccupationMatchRequest,
    OccupationMatchResponse,
)
from app.services import occupation_matcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/occupation", tags=["Occupation Matcher"])


@router.post(
    "/match",
    response_model=OccupationMatchResponse,
    summary="Match PEO JTK ke ESCO Occupation",
    description=(
        "Menerima teks PEO (Program Educational Objective) / Profil Lulusan JTK, "
        "mengekstrak vektor semantic SBERT, lalu menghitung Cosine Similarity "
        "terhadap seluruh ESCO Occupation untuk mendapatkan kandidat pekerjaan "
        "yang paling relevan (top-K)."
    ),
)
async def match_occupation(request: OccupationMatchRequest):
    """POST /occupation/match — Pencocokan PEO ke ESCO Occupation."""
    try:
        matches = occupation_matcher.match_peo_to_occupations(
            peo_text=request.peo_text,
            top_k=request.top_k,
        )
        return OccupationMatchResponse(
            peo_text=request.peo_text,
            matches=matches,
        )
    except Exception as e:
        logger.error("Error di /occupation/match: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
