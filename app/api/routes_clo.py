"""
API Routes — CLO (Step E→K default).

Endpoint:
- POST /clo/match-from-plo — Match PLO ke CLO menggunakan SBERT (default).
"""

import logging

from fastapi import APIRouter, HTTPException

from app.models.schemas import CLOMatchRequest, CLOMatchResponse
from app.services import clo_mapper

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/clo", tags=["CLO Mapping (E→K)"])


@router.post(
    "/match-from-plo",
    response_model=CLOMatchResponse,
    summary="Step E→K: Match PLO ke CLO (SBERT default)",
    description=(
        "Menerima ID PLO JTK, encode teks PLO dan semua CLO via SBERT, "
        "hitung cosine similarity, return top-K CLO yang relevan "
        "beserta info mata kuliah terkait (step K→L)."
    ),
)
async def match_clo_from_plo(request: CLOMatchRequest):
    """POST /clo/match-from-plo — Step E→K."""
    try:
        result = clo_mapper.match_plo_to_clo(
            plo_id=request.plo_id,
            top_k=request.top_k,
            clo_source=request.clo_source,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error("Error di /clo/match-from-plo: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
