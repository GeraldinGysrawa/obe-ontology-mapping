"""
Routes Integration — End-to-end integration pemetaan kurikulum (PEO, ESCO, PLO, CLO, Mata Kuliah, & CSO).

Endpoint:
- POST /integration/peo-plo-clo — End-to-End Integration PEO → PLO → CLO (dengan ESCO & CSO)
"""

import logging
import time

from fastapi import APIRouter

from app.core.ner import extract_skills_ner
from app.models.cso_schemas import (
    CloToTopicsResult,
    TopicMatch,
)
from app.models.schemas import (
    PipelineMeta,
    PipelineRequest,
    PipelineResponse,
    SkillDetail,
)
from app.repositories import jtk_repository
from app.repositories.cso_repository import CSORepository
from app.services import clo_mapper, occupation_matcher, skill_comparator, skill_lookup
from app.services.cso_topic_matcher import CSOTopicMatcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/integration", tags=["End-to-End Integration"])

cso_repo = CSORepository()
cso_matcher = CSOTopicMatcher()


async def _run_base_integration(request: PipelineRequest) -> PipelineResponse:
    """Jalankan integrasi dasar: PEO → ESCO Occupation → ESCO Skills vs PLO → CLO → Mata Kuliah."""
    errors = []
    meta = PipelineMeta(execution_time_ms={})

    response = PipelineResponse(
        peo=request.peo_name,
        meta=meta,
        errors=errors,
    )

    def get_ms(start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000, 2)

    # 1. Match PEO → ESCO Occupation
    start = time.perf_counter()
    try:
        matches = occupation_matcher.match_peo_to_occupations(
            peo_text=request.peo_name,
            top_k=1,
        )
        if matches:
            response.occupation = matches[0]
        else:
            errors.append("Tidak ada ESCO Occupation yang cocok untuk PEO ini.")
    except Exception as e:
        logger.error("Integration - Error saat match_peo_to_occupations: %s", str(e))
        errors.append(f"Gagal mencari ESCO Occupation: {str(e)}")

    meta.execution_time_ms["occupation_match"] = get_ms(start)

    if not response.occupation:
        errors.append("Skipping ESCO Skills lookup dan Comparison karena occupation tidak ditemukan.")
    else:
        # 2. Ambil ESCO Skills
        start = time.perf_counter()
        try:
            skills_raw = skill_lookup.get_skills_for_occupation(response.occupation.uri)
            response.esco_skills = [
                SkillDetail(
                    uri=s["uri"],
                    label=s["label"],
                    skill_type=s["skill_type"],
                    relation_type=s["relation_type"],
                    description=s.get("description"),
                )
                for s in skills_raw
            ]
            if not response.esco_skills:
                errors.append("ESCO Occupation ini tidak memiliki data skills.")
        except Exception as e:
            logger.error("Integration - Error saat get_skills_for_occupation: %s", str(e))
            errors.append(f"Gagal mengambil ESCO Skills: {str(e)}")
            response.esco_skills = []

        meta.execution_time_ms["skill_lookup"] = get_ms(start)

    # 3. Ambil PLO JTK terkait PEO
    start = time.perf_counter()
    peo_id = None
    try:
        peos = jtk_repository.get_peo_list()
        best_peo = None
        for p in peos:
            if request.peo_name.lower() in p.peo_text.lower():
                best_peo = p
                break

        if not best_peo and peos:
            from rapidfuzz import fuzz
            best_peo = max(peos, key=lambda p: fuzz.partial_ratio(request.peo_name.lower(), p.peo_text.lower()))
            if fuzz.partial_ratio(request.peo_name.lower(), best_peo.peo_text.lower()) < 50:
                best_peo = None

        if best_peo:
            peo_id = best_peo.peo_id
            response.plo_jtk = jtk_repository.get_plo_list(peo_id=peo_id)
            if not response.plo_jtk:
                errors.append(f"Tidak ada PLO ditemukan untuk PEO ID '{peo_id}'.")
        else:
            errors.append(f"Tidak dapat menemukan PEO ID untuk nama '{request.peo_name}' di data kurikulum.")
    except Exception as e:
        logger.error("Integration - Error saat mengambil PLO JTK: %s", str(e))
        errors.append(f"Gagal mengambil PLO JTK: {str(e)}")

    meta.execution_time_ms["plo_lookup"] = get_ms(start)

    # 4. Perbandingan ESCO Skills vs PLO JTK
    if response.esco_skills and response.plo_jtk:
        start = time.perf_counter()
        try:
            esco_skills_dicts = [
                {"uri": s.uri, "label": s.label, "description": s.description or ""}
                for s in response.esco_skills
            ]
            comp_result = skill_comparator.compare_skills_vs_plo(
                esco_skills=esco_skills_dicts,
                plo_list=response.plo_jtk,
                threshold=request.threshold,
            )
            response.covered = comp_result.covered
            response.gap = comp_result.gap
            response.overskill = comp_result.overskill
        except Exception as e:
            logger.error("Integration - Error saat compare_skills_vs_plo: %s", str(e))
            errors.append(f"Gagal membandingkan Skills vs PLO: {str(e)}")

        meta.execution_time_ms["comparison"] = get_ms(start)
    else:
        if not response.esco_skills and not any("ESCO Occupation" in err for err in errors):
            errors.append("Skipping Comparison karena data ESCO Skills kosong.")
        if not response.plo_jtk:
            errors.append("Skipping Comparison karena data PLO JTK kosong.")
        meta.execution_time_ms["comparison"] = 0.0

    # 5. Match PLO ke CLO dan dapatkan Mata Kuliah
    if response.plo_jtk:
        start = time.perf_counter()
        response.clo = []
        mk_set = set()

        try:
            for plo in response.plo_jtk:
                match_result = clo_mapper.match_plo_to_clo(
                    plo_id=plo.plo_id,
                    top_k=3,
                    clo_source="jtk",
                )
                response.clo.append(match_result)
                for clo_match in match_result.matched_clo:
                    if clo_match.mata_kuliah and clo_match.mata_kuliah.nama:
                        mk_set.add(clo_match.mata_kuliah.nama)

            response.mata_kuliah = sorted(list(mk_set))

        except Exception as e:
            logger.error("Integration - Error saat mapping CLO: %s", str(e))
            errors.append(f"Gagal melakukan mapping CLO: {str(e)}")

        meta.execution_time_ms["clo_mapping"] = get_ms(start)
    else:
        errors.append("Skipping CLO Mapping karena data PLO JTK kosong.")
        meta.execution_time_ms["clo_mapping"] = 0.0

    return response


def _collect_unique_clo_ids(response: PipelineResponse) -> list[str]:
    """Kumpulkan CLO ID unik dari hasil integrasi."""
    clo_ids = set()
    if response.clo:
        for clo_resp in response.clo:
            for clo_match in clo_resp.matched_clo:
                clo_ids.add(clo_match.clo_id)
    return sorted(clo_ids)


async def _extract_and_match_clo(clo_text: str, top_k_per_keyword: int = 1) -> tuple[list[str], list[TopicMatch]]:
    """NER lalu match keyword ke CSO topics."""
    try:
        keywords = await extract_skills_ner(clo_text)
    except Exception as e:
        logger.error(f"NER extraction failed: {e}")
        keywords = []

    if not keywords:
        return [], []

    all_matched: list[TopicMatch] = []
    seen_uris = set()

    for kw in keywords:
        matches = cso_matcher.match_keyword_to_topic(kw, top_k=top_k_per_keyword)
        for m in matches:
            if m.topic_uri not in seen_uris:
                all_matched.append(m)
                seen_uris.add(m.topic_uri)

    return keywords, all_matched


@router.post(
    "/peo-plo-clo",
    summary="End-to-End Integration PEO → PLO → CLO",
    description=(
        "Menjalankan integrasi pemetaan kurikulum secara end-to-end dimulai dari "
        "nama PEO (pencocokan PEO ke ESCO Occupation, pencarian ESCO Skills, "
        "analisis keselarasan ESCO Skills vs PLO JTK, serta pemetaan PLO ke CLO dan "
        "Mata Kuliah), dilanjutkan dengan ekstraksi kata kunci teknis via NER "
        "dan pemetaan ke topik-topik baku Computer Science Ontology (CSO) pada setiap CLO."
    ),
)
async def run_end_to_end_integration(request: PipelineRequest):
    """POST /integration/peo-plo-clo."""

    # 1. Jalankan integrasi dasar
    response = await _run_base_integration(request)

    # 2. Kumpulkan CLO unik dari hasil integrasi
    clo_ids = _collect_unique_clo_ids(response)

    if not clo_ids:
        return {
            "integration": response,
            "cso_topics": [],
            "cso_errors": ["Tidak ada CLO ditemukan dari integrasi, skip CSO."],
        }

    # 3. Untuk setiap CLO, jalankan NER → CSO
    start = time.perf_counter()
    cso_results: list[CloToTopicsResult] = []
    cso_errors: list[str] = []

    for clo_id in clo_ids:
        clo = jtk_repository.get_clo_by_id(clo_id)
        if not clo:
            cso_errors.append(f"CLO '{clo_id}' tidak ditemukan di repository.")
            continue

        try:
            keywords, matched_topics = await _extract_and_match_clo(clo.clo_text)
            cso_results.append(CloToTopicsResult(
                clo_id=clo_id,
                mata_kuliah=clo.mata_kuliah,
                clo_text=clo.clo_text,
                extracted_keywords=keywords,
                matched_topics=matched_topics,
            ))
        except Exception as e:
            logger.error(f"CSO mapping failed for CLO {clo_id}: {e}")
            cso_errors.append(f"Gagal mapping CSO untuk CLO '{clo_id}': {str(e)}")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.meta.execution_time_ms["cso_topic_matching"] = elapsed_ms

    logger.info(
        f"End-to-End Integration selesai. {len(cso_results)} CLO di-map ke CSO "
        f"dalam {elapsed_ms:.1f} ms."
    )

    return {
        "integration": response,
        "cso_topics": cso_results,
        "cso_errors": cso_errors,
    }
