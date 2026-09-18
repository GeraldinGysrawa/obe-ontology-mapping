"""
Routes Pipeline+CSO — End-to-end pipeline yang diperluas dengan modul CSO.

Endpoint baru ini MENIRU alur /pipeline/run secara internal,
lalu menambahkan tahap CSO di ujung untuk setiap CLO yang ditemukan.

Endpoint:
1. POST /pipeline-cso/run-topics
   → Pipeline A→C→D, A→E, D↔E→F/G, E→K→L + K→NER→CSO Topics (TANPA ESCO)
2. POST /pipeline-cso/run-gap
   → Pipeline A→C→D, A→E, D↔E→F/G, E→K→L + K→NER→CSO→ESCO Gap Analysis

CATATAN: /pipeline/run TIDAK disentuh sama sekali.
"""

import logging
import time

import numpy as np
from fastapi import APIRouter, Query
from rapidfuzz import fuzz

from app.config import get_settings
from app.core import embeddings
from app.core.ner import extract_skills_ner
from app.models.cso_schemas import (
    CloToTopicsResult,
    CloEscoGapResult,
    CsoEscoCovered,
    CsoEscoGap,
    TopicMatch,
)
from app.models.schemas import (
    PipelineMeta,
    PipelineRequest,
    PipelineResponse,
    SkillDetail,
)
from app.repositories import esco_repository, jtk_repository
from app.repositories.cso_repository import CSORepository
from app.services import clo_mapper, occupation_matcher, skill_comparator, skill_lookup
from app.services.cso_topic_matcher import CSOTopicMatcher

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline-cso", tags=["Pipeline + CSO"])

cso_repo = CSORepository()
cso_matcher = CSOTopicMatcher()


# ── Helper: jalankan pipeline utama (clone dari /pipeline/run) ────────────

async def _run_base_pipeline(request: PipelineRequest) -> PipelineResponse:
    """Jalankan pipeline utama A→C→D, A→E, D↔E→F/G, E→K→L.

    Ini adalah COPY PERSIS dari logic /pipeline/run, supaya /pipeline/run
    tidak tersentuh sama sekali.
    """
    errors = []
    meta = PipelineMeta(execution_time_ms={})

    response = PipelineResponse(
        peo=request.peo_name,
        meta=meta,
        errors=errors,
    )

    def get_ms(start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000, 2)

    # 1. Step A→C: PEO → ESCO Occupation
    start = time.perf_counter()
    try:
        matches = occupation_matcher.match_peo_to_occupations(
            peo_text=request.peo_name,
            top_k=request.top_k_occupation,
        )
        if matches:
            response.occupation = matches[0]
        else:
            errors.append("Tidak ada ESCO Occupation yang cocok untuk PEO ini.")
    except Exception as e:
        logger.error("Pipeline-CSO - Error saat match_peo_to_occupations: %s", str(e))
        errors.append(f"Gagal mencari ESCO Occupation: {str(e)}")

    meta.execution_time_ms["occupation_match"] = get_ms(start)

    if not response.occupation:
        errors.append("Skipping ESCO Skills lookup dan Comparison karena occupation tidak ditemukan.")
    else:
        # 2. Step C→D: Ambil ESCO Skills
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
            logger.error("Pipeline-CSO - Error saat get_skills_for_occupation: %s", str(e))
            errors.append(f"Gagal mengambil ESCO Skills: {str(e)}")
            response.esco_skills = []

        meta.execution_time_ms["skill_lookup"] = get_ms(start)

    # 3. Step A→E: Ambil PLO JTK terkait PEO
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
        logger.error("Pipeline-CSO - Error saat mengambil PLO JTK: %s", str(e))
        errors.append(f"Gagal mengambil PLO JTK: {str(e)}")

    meta.execution_time_ms["plo_lookup"] = get_ms(start)

    # 4. Step D↔E → F/G: Compare ESCO Skills vs PLO JTK
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
            logger.error("Pipeline-CSO - Error saat compare_skills_vs_plo: %s", str(e))
            errors.append(f"Gagal membandingkan Skills vs PLO: {str(e)}")

        meta.execution_time_ms["comparison"] = get_ms(start)
    else:
        if not response.esco_skills and not any("ESCO Occupation" in err for err in errors):
            errors.append("Skipping Comparison karena data ESCO Skills kosong.")
        if not response.plo_jtk:
            errors.append("Skipping Comparison karena data PLO JTK kosong.")
        meta.execution_time_ms["comparison"] = 0.0

    # 5. Step E→K→L: Match PLO ke CLO dan dapatkan Mata Kuliah
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
            logger.error("Pipeline-CSO - Error saat mapping CLO: %s", str(e))
            errors.append(f"Gagal melakukan mapping CLO: {str(e)}")

        meta.execution_time_ms["clo_mapping"] = get_ms(start)
    else:
        errors.append("Skipping CLO Mapping karena data PLO JTK kosong.")
        meta.execution_time_ms["clo_mapping"] = 0.0

    return response


# ── Helper: kumpulkan CLO ID unik dari hasil pipeline ─────────────────────

def _collect_unique_clo_ids(response: PipelineResponse) -> list[str]:
    """Kumpulkan CLO ID unik dari hasil pipeline E→K."""
    clo_ids = set()
    if response.clo:
        for clo_resp in response.clo:
            for clo_match in clo_resp.matched_clo:
                clo_ids.add(clo_match.clo_id)
    return sorted(clo_ids)


# ── Helper: NER + CSO matching untuk satu CLO ─────────────────────────────

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


# ── Helper: CSO matching TANPA NER untuk satu CLO ─────────────────────────

async def _match_no_ner_clo(clo_text: str, top_k_per_keyword: int = 3) -> tuple[list[str], list[TopicMatch]]:
    """Langsung match teks CLO ke CSO topics tanpa dipecah via NER."""
    keywords = [clo_text]
    all_matched: list[TopicMatch] = []
    seen_uris = set()

    for kw in keywords:
        matches = cso_matcher.match_keyword_to_topic(kw, top_k=top_k_per_keyword)
        for m in matches:
            if m.topic_uri not in seen_uris:
                all_matched.append(m)
                seen_uris.add(m.topic_uri)

    return keywords, all_matched


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 1: Pipeline + CSO Topics (TANPA perbandingan ESCO)
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/run-topics",
    summary="Pipeline end-to-end + CSO Topics (tanpa ESCO)",
    description=(
        "Menjalankan pipeline utama A→C→D, A→E, D↔E→F/G, E→K→L, "
        "lalu menambahkan tahap NER→CSO Topics untuk setiap CLO yang ditemukan. "
        "Hasil CSO TIDAK dibandingkan ke ESCO."
    ),
)
async def run_pipeline_with_cso_topics(request: PipelineRequest):
    """POST /pipeline-cso/run-topics."""

    # 1. Jalankan pipeline utama
    response = await _run_base_pipeline(request)

    # 2. Kumpulkan CLO unik dari hasil pipeline
    clo_ids = _collect_unique_clo_ids(response)

    if not clo_ids:
        return {
            "pipeline": response,
            "cso_topics": [],
            "cso_errors": ["Tidak ada CLO ditemukan dari pipeline, skip CSO."],
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
        f"Pipeline+CSO Topics selesai. {len(cso_results)} CLO di-map ke CSO "
        f"dalam {elapsed_ms:.1f} ms."
    )

    return {
        "pipeline": response,
        "cso_topics": cso_results,
        "cso_errors": cso_errors,
    }


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 1B: Pipeline + CSO Topics (TANPA NER, tanpa ESCO)
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/run-topics-no-ner",
    summary="Pipeline end-to-end + CSO Topics (tanpa ESCO) TANPA NER",
    description=(
        "Menjalankan pipeline utama A→C→D, A→E, D↔E→F/G, E→K→L, "
        "lalu menambahkan tahap pencocokan teks utuh CLO ke CSO Topics. "
        "TANPA NER. Hasil CSO TIDAK dibandingkan ke ESCO."
    ),
)
async def run_pipeline_with_cso_topics_no_ner(request: PipelineRequest):
    """POST /pipeline-cso/run-topics-no-ner."""

    # 1. Jalankan pipeline utama
    response = await _run_base_pipeline(request)

    # 2. Kumpulkan CLO unik dari hasil pipeline
    clo_ids = _collect_unique_clo_ids(response)

    if not clo_ids:
        return {
            "pipeline": response,
            "cso_topics": [],
            "cso_errors": ["Tidak ada CLO ditemukan dari pipeline, skip CSO."],
        }

    # 3. Untuk setiap CLO, jalankan NO NER → CSO
    start = time.perf_counter()
    cso_results: list[CloToTopicsResult] = []
    cso_errors: list[str] = []

    for clo_id in clo_ids:
        clo = jtk_repository.get_clo_by_id(clo_id)
        if not clo:
            cso_errors.append(f"CLO '{clo_id}' tidak ditemukan di repository.")
            continue

        try:
            keywords, matched_topics = await _match_no_ner_clo(clo.clo_text)
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
        f"Pipeline+CSO Topics (No NER) selesai. {len(cso_results)} CLO di-map ke CSO "
        f"dalam {elapsed_ms:.1f} ms."
    )

    return {
        "pipeline": response,
        "cso_topics": cso_results,
        "cso_errors": cso_errors,
    }


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 2: Pipeline + CSO → ESCO Gap Analysis
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/run-gap",
    summary="Pipeline end-to-end + CSO → ESCO Gap Analysis",
    description=(
        "Menjalankan pipeline utama A→C→D, A→E, D↔E→F/G, E→K→L, "
        "lalu menambahkan tahap NER→CSO→SBERT+Cosine→ESCO untuk setiap CLO. "
        "Menghasilkan klasifikasi Tercakup (P) dan Gap (Q) per CLO."
    ),
)
async def run_pipeline_with_cso_gap(
    request: PipelineRequest,
    threshold: float = Query(None, description="Threshold similarity CSO↔ESCO. Jika None, dianggap 0.0 (tampilkan semua)"),
):
    """POST /pipeline-cso/run-gap."""

    settings = get_settings()
    if threshold is None:
        threshold = 0.0  # Tampilkan semua jika tidak diisi

    # 1. Jalankan pipeline utama
    response = await _run_base_pipeline(request)

    # 2. Kumpulkan CLO unik
    clo_ids = _collect_unique_clo_ids(response)

    if not clo_ids:
        return {
            "pipeline": response,
            "cso_gap_analysis": [],
            "cso_errors": ["Tidak ada CLO ditemukan dari pipeline, skip CSO."],
        }

    # 3. Pre-load ESCO embeddings (sekali untuk semua CLO)
    esco_data = esco_repository.get_all_skill_texts()
    esco_texts = [f"{label}. {desc}" for _, label, desc in esco_data]
    esco_emb = embeddings.load_or_compute_embeddings(esco_texts, cache_key="esco_skills")

    # 4. Untuk setiap CLO, jalankan NER → CSO → ESCO Gap
    start = time.perf_counter()
    gap_results: list[CloEscoGapResult] = []
    cso_errors: list[str] = []

    for clo_id in clo_ids:
        clo = jtk_repository.get_clo_by_id(clo_id)
        if not clo:
            cso_errors.append(f"CLO '{clo_id}' tidak ditemukan di repository.")
            continue

        try:
            keywords, matched_topics = await _extract_and_match_clo(clo.clo_text)

            if not matched_topics:
                gap_results.append(CloEscoGapResult(
                    clo_id=clo_id,
                    mata_kuliah=clo.mata_kuliah,
                    clo_text=clo.clo_text,
                    extracted_keywords=keywords,
                    matched_topics=[],
                    threshold_used=threshold,
                    total_cso_topics=0,
                    total_covered=0,
                    total_gap=0,
                ))
                continue

            # Encode CSO topics
            topic_labels = [m.topic_name for m in matched_topics]
            cso_emb = embeddings.encode(topic_labels, clean=False)

            # Cosine similarity matrix
            sim_matrix = embeddings.cosine_sim(cso_emb, esco_emb)

            # Klasifikasi per topik CSO
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

            gap_results.append(CloEscoGapResult(
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
            ))

        except Exception as e:
            logger.error(f"CSO gap analysis failed for CLO {clo_id}: {e}")
            cso_errors.append(f"Gagal gap analysis CSO untuk CLO '{clo_id}': {str(e)}")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.meta.execution_time_ms["cso_gap_analysis"] = elapsed_ms

    logger.info(
        f"Pipeline+CSO Gap selesai. {len(gap_results)} CLO dianalisis "
        f"dalam {elapsed_ms:.1f} ms."
    )

    return {
        "pipeline": response,
        "cso_gap_analysis": gap_results,
        "cso_errors": cso_errors,
    }


# ══════════════════════════════════════════════════════════════════════════
# ENDPOINT 2B: Pipeline + CSO → ESCO Gap Analysis (TANPA NER)
# ══════════════════════════════════════════════════════════════════════════

@router.post(
    "/run-gap-no-ner",
    summary="Pipeline end-to-end + CSO → ESCO Gap Analysis TANPA NER",
    description=(
        "Menjalankan pipeline utama A→C→D, A→E, D↔E→F/G, E→K→L, "
        "lalu menambahkan tahap teks utuh CLO→CSO→SBERT+Cosine→ESCO untuk setiap CLO. "
        "TANPA NER. Menghasilkan klasifikasi Tercakup (P) dan Gap (Q) per CLO."
    ),
)
async def run_pipeline_with_cso_gap_no_ner(
    request: PipelineRequest,
    threshold: float = Query(None, description="Threshold similarity CSO↔ESCO. Jika None, dianggap 0.0 (tampilkan semua)"),
):
    """POST /pipeline-cso/run-gap-no-ner."""

    settings = get_settings()
    if threshold is None:
        threshold = 0.0  # Tampilkan semua jika tidak diisi

    # 1. Jalankan pipeline utama
    response = await _run_base_pipeline(request)

    # 2. Kumpulkan CLO unik
    clo_ids = _collect_unique_clo_ids(response)

    if not clo_ids:
        return {
            "pipeline": response,
            "cso_gap_analysis": [],
            "cso_errors": ["Tidak ada CLO ditemukan dari pipeline, skip CSO."],
        }

    # 3. Pre-load ESCO embeddings (sekali untuk semua CLO)
    esco_data = esco_repository.get_all_skill_texts()
    esco_texts = [f"{label}. {desc}" for _, label, desc in esco_data]
    esco_emb = embeddings.load_or_compute_embeddings(esco_texts, cache_key="esco_skills")

    # 4. Untuk setiap CLO, jalankan NO NER → CSO → ESCO Gap
    start = time.perf_counter()
    gap_results: list[CloEscoGapResult] = []
    cso_errors: list[str] = []

    for clo_id in clo_ids:
        clo = jtk_repository.get_clo_by_id(clo_id)
        if not clo:
            cso_errors.append(f"CLO '{clo_id}' tidak ditemukan di repository.")
            continue

        try:
            keywords, matched_topics = await _match_no_ner_clo(clo.clo_text)

            if not matched_topics:
                gap_results.append(CloEscoGapResult(
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
                ))
                continue

            # Encode CSO topics
            topic_labels = [m.topic_name for m in matched_topics]
            cso_emb = embeddings.encode(topic_labels, clean=False)

            # Cosine similarity matrix
            sim_matrix = embeddings.cosine_sim(cso_emb, esco_emb)

            # Klasifikasi per topik CSO
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

            gap_results.append(CloEscoGapResult(
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
            ))

        except Exception as e:
            logger.error(f"CSO gap analysis (No NER) failed for CLO {clo_id}: {e}")
            cso_errors.append(f"Gagal gap analysis CSO (No NER) untuk CLO '{clo_id}': {str(e)}")

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
    response.meta.execution_time_ms["cso_gap_analysis"] = elapsed_ms

    logger.info(
        f"Pipeline+CSO Gap (No NER) selesai. {len(gap_results)} CLO dianalisis "
        f"dalam {elapsed_ms:.1f} ms."
    )

    return {
        "pipeline": response,
        "cso_gap_analysis": gap_results,
        "cso_errors": cso_errors,
    }

