"""
Routes CSO — Modul jembatan CLO ke Computer Science Ontology (CSO).

Endpoint:
- POST /cso/clo-to-topics — CLO → NER → CSO Topics
"""

import logging
from fastapi import APIRouter, HTTPException

from app.core.ner import extract_skills_ner
from app.models.cso_schemas import CloToTopicsRequest, CloToTopicsResult, TopicMatch
from app.repositories.cso_repository import CSORepository
from app.repositories.jtk_repository import get_clo_by_id
from app.services.cso_topic_matcher import CSOTopicMatcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cso", tags=["CSO Module"])

cso_repo = CSORepository()
cso_matcher = CSOTopicMatcher()


def _load_clo(clo_id: str):
    """Load CLO dari jtk_repository, raise HTTPException jika gagal."""
    clo = get_clo_by_id(clo_id)
    if not clo:
        raise HTTPException(status_code=404, detail=f"CLO '{clo_id}' not found.")
    return clo


async def _extract_and_match(clo_text: str, top_k_per_keyword: int = 1) -> tuple[list[str], list[TopicMatch]]:
    """Jalankan NER lalu match keyword ke CSO topics."""
    try:
        keywords = await extract_skills_ner(clo_text)
    except Exception as e:
        logger.error(f"NER extraction failed: {e}")
        keywords = []

    if not keywords:
        return [], []

    logger.info(f"NER extracted {len(keywords)} keywords: {keywords}")

    all_matched: list[TopicMatch] = []
    seen_uris = set()

    for kw in keywords:
        matches = cso_matcher.match_keyword_to_topic(kw, top_k=top_k_per_keyword)
        for m in matches:
            if m.topic_uri not in seen_uris:
                all_matched.append(m)
                seen_uris.add(m.topic_uri)

    logger.info(f"Matched {len(all_matched)} unique CSO topics.")
    return keywords, all_matched


@router.post(
    "/clo-to-topics",
    response_model=CloToTopicsResult,
    summary="Pemetaan Kata Kunci CLO ke CSO Topics",
    description=(
        "Mengekstraksi kata kunci teknis dari teks Course Learning Outcome (CLO) berdasarkan clo_id "
        "menggunakan Named Entity Recognition (NER), lalu mencocokkan kata kunci "
        "tersebut ke topik-topik baku pada Computer Science Ontology (CSO)."
    ),
)
async def map_clo_to_topics(request: CloToTopicsRequest):
    """POST /cso/clo-to-topics — Pemetaan CLO ke CSO Topics."""
    if not cso_repo._is_loaded:
        raise HTTPException(status_code=503, detail="CSO Graph belum ter-load.")

    clo = _load_clo(request.clo_id)
    keywords, matched_topics = await _extract_and_match(clo.clo_text, top_k_per_keyword=1)

    return CloToTopicsResult(
        clo_id=request.clo_id,
        mata_kuliah=clo.mata_kuliah,
        clo_text=clo.clo_text,
        extracted_keywords=keywords,
        matched_topics=matched_topics,
    )
