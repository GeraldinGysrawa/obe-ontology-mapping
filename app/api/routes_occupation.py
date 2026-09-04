"""
API Routes — Occupation (Step A→C, C→D).

Endpoint:
- POST /occupation/match — PEO JTK → top-K ESCO Occupation (SBERT+Cosine)
- GET  /occupation/skills — ESCO Skills untuk satu occupation (lookup langsung)
"""

import logging

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    OccupationMatchRequest,
    OccupationMatchResponse,
    OccupationSkillsResponse,
    SkillDetail,
)
from app.repositories import esco_repository
from app.services import occupation_matcher, skill_lookup

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/occupation", tags=["Occupation (A→C→D)"])


@router.post(
    "/match",
    response_model=OccupationMatchResponse,
    summary="Step A→C: Match PEO ke ESCO Occupation",
    description=(
        "Menerima teks PEO/profil lulusan JTK, encode via SBERT, "
        "hitung cosine similarity terhadap semua ESCO Occupation, "
        "return top-K hasil."
    ),
)
async def match_occupation(request: OccupationMatchRequest):
    """POST /occupation/match — Step A→C."""
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


@router.get(
    "/skills",
    response_model=OccupationSkillsResponse,
    summary="Step C→D: Ambil ESCO Skills untuk occupation",
    description=(
        "Lookup langsung dari occupationSkillRelations — "
        "BUKAN similarity, melainkan relasi resmi ESCO."
    ),
)
async def get_occupation_skills(
    uri: str = Query(
        ...,
        description="ESCO Occupation URI (mis. http://data.europa.eu/esco/occupation/...)",
    ),
):
    """GET /occupation/skills?uri=... — Step C→D."""
    # Cek occupation exists
    occ = esco_repository.get_occupation_by_uri(uri)
    if occ is None:
        raise HTTPException(
            status_code=404,
            detail=f"Occupation dengan URI '{uri}' tidak ditemukan di data ESCO.",
        )

    skills_raw = skill_lookup.get_skills_for_occupation(uri)

    skills = [
        SkillDetail(
            uri=s["uri"],
            label=s["label"],
            skill_type=s["skill_type"],
            relation_type=s["relation_type"],
            description=s.get("description"),
        )
        for s in skills_raw
    ]

    return OccupationSkillsResponse(
        occupation_uri=uri,
        occupation_label=occ["label"],
        total_skills=len(skills),
        skills=skills,
    )
