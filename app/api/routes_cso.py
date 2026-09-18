"""
Routes CSO — Modul jembatan CLO ↔ ESCO via Computer Science Ontology.

Endpoint:
1. POST /cso/clo-to-topics/{clo_id}  — CLO → NER → CSO Topics (tanpa ESCO)
2. POST /cso/clo-to-esco/{clo_id}    — CLO → NER → CSO Topics → SBERT+Cosine → ESCO Gap Analysis
3. GET  /cso/topic-search?query=...   — Eksplorasi manual topik CSO
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Literal, List
import logging
import numpy as np

from app.models.cso_schemas import (
    TopicMatch,
    CloToTopicsResult,
    CloEscoGapResult,
    CsoEscoCovered,
    CsoEscoGap,
    CloToEscoViaCsoResult,
)
from app.repositories.cso_repository import CSORepository
from app.repositories.jtk_repository import get_clo_by_id
from app.services.cso_topic_matcher import CSOTopicMatcher
from app.services.cso_graph_expander import CSOGraphExpander
from app.core.ner import extract_skills_ner
from app.core import embeddings
from app.repositories import esco_repository
from app.config import get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/cso", tags=["CSO Module"])

cso_repo = CSORepository()
cso_matcher = CSOTopicMatcher()
cso_expander = CSOGraphExpander()


# ── Helper: load CLO ──────────────────────────────────────────────────────

def _load_clo(clo_id: str):
    """Load CLO dari jtk_repository, raise HTTPException jika gagal."""
    clo = get_clo_by_id(clo_id)
    if not clo:
        raise HTTPException(status_code=404, detail=f"CLO '{clo_id}' not found.")
    return clo


# ── Helper: NER + CSO matching (shared oleh kedua endpoint) ───────────────

async def _extract_and_match(clo_text: str, top_k_per_keyword: int = 1) -> tuple[list[str], list[TopicMatch]]:
    """Jalankan NER lalu match keyword ke CSO topics."""
    # 1. NER
    try:
        keywords = await extract_skills_ner(clo_text)
    except Exception as e:
        logger.error(f"NER extraction failed: {e}")
        keywords = []

    if not keywords:
        return [], []

    logger.info(f"NER extracted {len(keywords)} keywords: {keywords}")

    # 2. Match ke CSO Topics
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


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 1: CLO → CSO Topics (TANPA ESCO)
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/clo-to-topics/{clo_id}",
    response_model=CloToTopicsResult,
    summary="CLO → CSO Topics (tanpa ESCO)",
)
async def map_clo_to_topics(
    clo_id: str,
    top_k_per_keyword: int = Query(1, ge=1, description="Top-K CSO topic per keyword"),
):
    """
    Ekstrak keyword dari teks CLO via NER, lalu cocokkan ke topik CSO baku.
    Tidak ada perbandingan ke ESCO — murni untuk melihat representasi CSO dari CLO.
    """
    if not cso_repo._is_loaded:
        raise HTTPException(status_code=503, detail="CSO Graph belum ter-load.")

    clo = _load_clo(clo_id)
    keywords, matched_topics = await _extract_and_match(clo.clo_text, top_k_per_keyword)

    return CloToTopicsResult(
        clo_id=clo_id,
        mata_kuliah=clo.mata_kuliah,
        clo_text=clo.clo_text,
        extracted_keywords=keywords,
        matched_topics=matched_topics,
    )


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 2: CLO → CSO → ESCO Gap Analysis
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/clo-to-esco/{clo_id}",
    response_model=CloEscoGapResult,
    summary="CLO → CSO → ESCO Gap Analysis",
)
async def map_clo_to_esco_gap(
    clo_id: str,
    threshold: float = Query(None, description="Threshold similarity. Jika None, dianggap 0.0 (tampilkan semua)"),
    top_k_per_keyword: int = Query(1, ge=1, description="Top-K CSO topic per keyword"),
):
    """
    Pipeline utuh: CLO → NER → CSO Topics → SBERT+Cosine langsung ke ESCO Skills.
    TANPA ekspansi graf (sesuai diagram revisi).
    Menghasilkan klasifikasi Tercakup (P) dan Gap (Q).
    """
    if not cso_repo._is_loaded:
        raise HTTPException(status_code=503, detail="CSO Graph belum ter-load.")

    settings = get_settings()
    if threshold is None:
        threshold = 0.0  # Tampilkan semua jika tidak diisi

    clo = _load_clo(clo_id)
    keywords, matched_topics = await _extract_and_match(clo.clo_text, top_k_per_keyword)

    if not matched_topics:
        return CloEscoGapResult(
            clo_id=clo_id,
            mata_kuliah=clo.mata_kuliah,
            clo_text=clo.clo_text,
            extracted_keywords=keywords,
            matched_topics=[],
            threshold_used=threshold,
            total_cso_topics=0,
            total_covered=0,
            total_gap=0,
        )

    # Ambil label topik CSO yang sudah di-match
    topic_labels = [m.topic_name for m in matched_topics]

    # Encode topik CSO
    cso_emb = embeddings.encode(topic_labels, clean=False)  # shape (T, dim)

    # Ambil ESCO Skills + embeddings (pakai cache)
    esco_data = esco_repository.get_all_skill_texts()  # list of (uri, label, desc)
    esco_texts = [f"{label}. {desc}" for _, label, desc in esco_data]
    esco_emb = embeddings.load_or_compute_embeddings(esco_texts, cache_key="esco_skills")

    # Cosine similarity matrix (T x S)
    sim_matrix = embeddings.cosine_sim(cso_emb, esco_emb)

    # Klasifikasi per topik CSO: covered (P) atau gap (Q)
    covered_list: list[CsoEscoCovered] = []
    gap_list: list[CsoEscoGap] = []

    for i, topic_label in enumerate(topic_labels):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])
        best_uri, best_label, _ = esco_data[best_idx]

        if best_score >= threshold:
            covered_list.append(CsoEscoCovered(
                cso_topic=topic_label,
                best_esco_skill_uri=best_uri,
                best_esco_skill_label=best_label,
                similarity_score=round(best_score, 4),
            ))
        else:
            gap_list.append(CsoEscoGap(
                cso_topic=topic_label,
                best_esco_skill_uri=best_uri,
                best_esco_skill_label=best_label,
                best_similarity_score=round(best_score, 4),
            ))

    logger.info(
        f"CLO {clo_id}: {len(covered_list)} covered, {len(gap_list)} gap "
        f"(threshold={threshold})"
    )

    return CloEscoGapResult(
        clo_id=clo_id,
        mata_kuliah=clo.mata_kuliah,
        clo_text=clo.clo_text,
        extracted_keywords=keywords,
        matched_topics=matched_topics,
        threshold_used=threshold,
        total_cso_topics=len(topic_labels),
        total_covered=len(covered_list),
        total_gap=len(gap_list),
        covered=covered_list,
        gap=gap_list,
    )



# ── Helper: CSO matching TANPA NER ──────────────────────────────────────────

async def _match_no_ner(clo_text: str, top_k_per_keyword: int = 3) -> tuple[list[str], list[TopicMatch]]:
    """Match seluruh teks CLO ke CSO topics tanpa memecahnya via NER."""
    # Anggap seluruh teks CLO sebagai 1 keyword panjang
    keywords = [clo_text]
    
    all_matched: list[TopicMatch] = []
    seen_uris = set()

    for kw in keywords:
        matches = cso_matcher.match_keyword_to_topic(kw, top_k=top_k_per_keyword)
        for m in matches:
            if m.topic_uri not in seen_uris:
                all_matched.append(m)
                seen_uris.add(m.topic_uri)

    logger.info(f"Matched {len(all_matched)} unique CSO topics (tanpa NER).")
    return keywords, all_matched


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 3: CLO → CSO → ESCO Gap Analysis (TANPA NER)
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/clo-to-esco-no-ner/{clo_id}",
    response_model=CloEscoGapResult,
    summary="CLO → CSO → ESCO Gap Analysis (TANPA NER)",
)
async def map_clo_to_esco_gap_no_ner(
    clo_id: str,
    threshold: float = Query(None, description="Threshold similarity. Jika None, dianggap 0.0 (tampilkan semua)"),
    top_k_per_keyword: int = Query(3, ge=1, description="Top-K CSO topic per keyword"),
):
    """
    Pipeline utuh: CLO (sebagai teks utuh) → CSO Topics → SBERT+Cosine langsung ke ESCO Skills.
    TANPA NER.
    """
    if not cso_repo._is_loaded:
        raise HTTPException(status_code=503, detail="CSO Graph belum ter-load.")

    settings = get_settings()
    if threshold is None:
        threshold = 0.0  # Tampilkan semua jika tidak diisi

    clo = _load_clo(clo_id)
    keywords, matched_topics = await _match_no_ner(clo.clo_text, top_k_per_keyword)

    if not matched_topics:
        return CloEscoGapResult(
            clo_id=clo_id,
            mata_kuliah=clo.mata_kuliah,
            clo_text=clo.clo_text,
            extracted_keywords=keywords,
            matched_topics=[],
            threshold_used=threshold,
            total_cso_topics=0,
            total_covered=0,
            total_gap=0,
            covered=[],
            gap=[],
        )

    # Ambil label topik CSO yang sudah di-match
    topic_labels = [m.topic_name for m in matched_topics]

    # Encode topik CSO
    cso_emb = embeddings.encode(topic_labels, clean=False)  # shape (T, dim)

    # Ambil ESCO Skills + embeddings (pakai cache)
    esco_data = esco_repository.get_all_skill_texts()  # list of (uri, label, desc)
    esco_texts = [f"{label}. {desc}" for _, label, desc in esco_data]
    esco_emb = embeddings.load_or_compute_embeddings(esco_texts, cache_key="esco_skills")

    # Cosine similarity matrix (T x S)
    sim_matrix = embeddings.cosine_sim(cso_emb, esco_emb)

    # Klasifikasi per topik CSO: covered (P) atau gap (Q)
    covered_list: list[CsoEscoCovered] = []
    gap_list: list[CsoEscoGap] = []

    for i, topic_label in enumerate(topic_labels):
        best_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_idx])
        best_uri, best_label, _ = esco_data[best_idx]

        if best_score >= threshold:
            covered_list.append(CsoEscoCovered(
                cso_topic=topic_label,
                best_esco_skill_uri=best_uri,
                best_esco_skill_label=best_label,
                similarity_score=round(best_score, 4),
            ))
        else:
            gap_list.append(CsoEscoGap(
                cso_topic=topic_label,
                best_esco_skill_uri=best_uri,
                best_esco_skill_label=best_label,
                best_similarity_score=round(best_score, 4),
            ))

    logger.info(
        f"CLO {clo_id} (No NER): {len(covered_list)} covered, {len(gap_list)} gap "
        f"(threshold={threshold})"
    )

    return CloEscoGapResult(
        clo_id=clo_id,
        mata_kuliah=clo.mata_kuliah,
        clo_text=clo.clo_text,
        extracted_keywords=keywords,
        matched_topics=matched_topics,
        threshold_used=threshold,
        total_cso_topics=len(topic_labels),
        total_covered=len(covered_list),
        total_gap=len(gap_list),
        covered=covered_list,
        gap=gap_list,
    )


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT Bantu: Topic Search (eksplorasi manual)
# ══════════════════════════════════════════════════════════════════════════

@router.get("/topic-search", summary="Eksplorasi manual topik CSO")
async def topic_search(
    query: str = Query(..., description="Keyword untuk dicari di CSO"),
    mode: Literal["hierarchy", "contribution"] = Query("hierarchy"),
    hop: int = Query(1, ge=0),
    direction: Literal["up", "down", "both"] = Query("up"),
    top_k_topics: int = Query(3),
):
    """
    Endpoint bantu untuk eksplorasi manual CSO topics dan graph expansion.
    """
    if not cso_repo._is_loaded:
        raise HTTPException(status_code=503, detail="CSO Graph belum ter-load.")

    matched_topics = cso_matcher.match_keyword_to_topic(query, top_k=top_k_topics)

    # Opsional: expand topics
    all_expanded_labels = set()
    expansion_details = {}

    for match in matched_topics:
        expanded = cso_expander.expand_topic(
            match.topic_uri, mode=mode, hop=hop, direction=direction
        )
        all_expanded_labels.update(expanded)
        expansion_details[match.topic_name] = expanded

    return {
        "query": query,
        "matched_topics": matched_topics,
        "expanded_topics_count": len(all_expanded_labels),
        "expanded_topics": list(all_expanded_labels),
        "expansion_details": expansion_details,
    }
