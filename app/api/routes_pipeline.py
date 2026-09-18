"""
API Routes — Orchestrator (Pipeline End-to-End).

Endpoint:
- POST /pipeline/run — Menjalankan seluruh tahap A→B→C→D, A→E, D↔E→F/G, E→K→L
  secara berurutan, dan mengumpulkan hasilnya dalam satu JSON response terstruktur
  beserta informasi waktu eksekusi tiap tahap (untuk keperluan laporan skripsi).
"""

import logging
import time

from fastapi import APIRouter, HTTPException
from rapidfuzz import fuzz

from app.models.schemas import (
    PipelineMeta,
    PipelineRequest,
    PipelineResponse,
    SkillDetail,
)
from app.repositories import jtk_repository
from app.services import clo_mapper, occupation_matcher, skill_comparator, skill_lookup

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["Pipeline Orchestrator"])


@router.post(
    "/run",
    response_model=PipelineResponse,
    summary="Menjalankan pipeline end-to-end secara berurutan",
    description="Menjalankan tahap A→C, C→D, A→E, D↔E→F/G, dan E→K→L dalam satu pemanggilan API.",
)
async def run_pipeline(request: PipelineRequest):
    """POST /pipeline/run."""
    errors = []
    meta = PipelineMeta(execution_time_ms={})

    # Response awal
    response = PipelineResponse(
        peo=request.peo_name,
        meta=meta,
        errors=errors,
    )

    # Helper function untuk format time
    def get_ms(start_time: float) -> float:
        return round((time.perf_counter() - start_time) * 1000, 2)

    # ────────────────────────────────────────────────────────────────────────
    # 1. Step A→C: PEO → ESCO Occupation
    # ────────────────────────────────────────────────────────────────────────
    start = time.perf_counter()
    try:
        matches = occupation_matcher.match_peo_to_occupations(
            peo_text=request.peo_name,
            top_k=request.top_k_occupation,
        )
        if matches:
            response.occupation = matches[0]  # Ambil yang teratas
        else:
            errors.append("Tidak ada ESCO Occupation yang cocok untuk PEO ini.")
    except Exception as e:
        logger.error("Pipeline - Error saat match_peo_to_occupations: %s", str(e))
        errors.append(f"Gagal mencari ESCO Occupation: {str(e)}")

    meta.execution_time_ms["occupation_match"] = get_ms(start)

    # Jika occupation tidak ketemu, tidak bisa lanjut cari skill ESCO (Step C→D dan D↔E)
    # Tapi masih bisa lanjut cari PLO (A→E)
    if not response.occupation:
        errors.append("Skipping ESCO Skills lookup dan Comparison karena occupation tidak ditemukan.")
    else:
        # ────────────────────────────────────────────────────────────────────────
        # 2. Step C→D: Ambil ESCO Skills untuk Occupation teratas
        # ────────────────────────────────────────────────────────────────────────
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
            logger.error("Pipeline - Error saat get_skills_for_occupation: %s", str(e))
            errors.append(f"Gagal mengambil ESCO Skills: {str(e)}")
            response.esco_skills = []

        meta.execution_time_ms["skill_lookup"] = get_ms(start)

    # ────────────────────────────────────────────────────────────────────────
    # 3. Step A→E: Ambil PLO JTK terkait PEO
    # ────────────────────────────────────────────────────────────────────────
    start = time.perf_counter()
    peo_id = None
    try:
        # Cari PEO ID berdasarkan nama (karena input hanya nama PEO)
        peos = jtk_repository.get_peo_list()
        
        # Simple text matching (bisa juga fuzzy kalau mau lebih robust)
        best_peo = None
        for p in peos:
            if request.peo_name.lower() in p.peo_text.lower():
                best_peo = p
                break
        
        # Fallback to fuzzy if exact substring fails
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
        logger.error("Pipeline - Error saat mengambil PLO JTK: %s", str(e))
        errors.append(f"Gagal mengambil PLO JTK: {str(e)}")

    meta.execution_time_ms["plo_lookup"] = get_ms(start)

    # ────────────────────────────────────────────────────────────────────────
    # 4. Step D↔E → F/G: Compare ESCO Skills vs PLO JTK
    # ────────────────────────────────────────────────────────────────────────
    if response.esco_skills and response.plo_jtk:
        start = time.perf_counter()
        try:
            # Konversi kembali format agar sesuai dengan service
            esco_skills_dicts = [
                {"uri": s.uri, "label": s.label, "description": s.description or ""}
                for s in response.esco_skills
            ]
            comp_result = skill_comparator.compare_skills_vs_plo(
                esco_skills=esco_skills_dicts,
                plo_list=response.plo_jtk,
                threshold=request.threshold,  # pakai dari parameter atau default .env
            )
            response.covered = comp_result.covered
            response.gap = comp_result.gap
            response.overskill = comp_result.overskill
        except Exception as e:
            logger.error("Pipeline - Error saat compare_skills_vs_plo: %s", str(e))
            errors.append(f"Gagal membandingkan Skills vs PLO: {str(e)}")
        
        meta.execution_time_ms["comparison"] = get_ms(start)
    else:
        if not response.esco_skills and not any("ESCO Occupation" in err for err in errors):
            errors.append("Skipping Comparison karena data ESCO Skills kosong.")
        if not response.plo_jtk:
            errors.append("Skipping Comparison karena data PLO JTK kosong.")
        meta.execution_time_ms["comparison"] = 0.0

    # ────────────────────────────────────────────────────────────────────────
    # 5. Step E→K→L: Match PLO ke CLO dan dapatkan Mata Kuliah
    # ────────────────────────────────────────────────────────────────────────
    if response.plo_jtk:
        start = time.perf_counter()
        response.clo = []
        mk_set = set()
        
        try:
            for plo in response.plo_jtk:
                # Gunakan metode sbert_only (default)
                match_result = clo_mapper.match_plo_to_clo(
                    plo_id=plo.plo_id,
                    top_k=3,
                    clo_source="jtk",
                )
                
                # Masukkan seluruh respons (yang berisi plo_id dan plo_text)
                response.clo.append(match_result)
                
                # Kumpulkan nama mata kuliah unik
                for clo_match in match_result.matched_clo:
                    if clo_match.mata_kuliah and clo_match.mata_kuliah.nama:
                        mk_set.add(clo_match.mata_kuliah.nama)
            
            # Sort mata_kuliah
            response.mata_kuliah = sorted(list(mk_set))
            
        except Exception as e:
            logger.error("Pipeline - Error saat mapping CLO: %s", str(e))
            errors.append(f"Gagal melakukan mapping CLO: {str(e)}")
            
        meta.execution_time_ms["clo_mapping"] = get_ms(start)
    else:
        errors.append("Skipping CLO Mapping karena data PLO JTK kosong.")
        meta.execution_time_ms["clo_mapping"] = 0.0

    return response
