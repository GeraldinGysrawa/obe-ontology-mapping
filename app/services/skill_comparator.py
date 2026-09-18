"""
Step D↔E: Hitung similarity ESCO Skills vs PLO JTK → F (covered) / G (gap).

Pipeline node: D (ESCO Skills) <--SBERT+Cosine--> E (PLO JTK)
               D + E --threshold--> F (skill ESCO tercakup)
                                    G (skill ESCO belum tercakup / gap)

Untuk tiap ESCO Skill: cari PLO dengan skor similarity tertinggi.
- Jika skor >= threshold → masuk F (covered)
- Jika skor < threshold  → masuk G (gap)
"""

import logging
import time

import numpy as np

from app.config import get_settings
from app.core import embeddings
from app.models.schemas import ComparisonResponse, CoveredSkill, GapSkill, Overskill, PLO

logger = logging.getLogger(__name__)


def compare_skills_vs_plo(
    esco_skills: list[dict],
    plo_list: list[PLO],
    threshold: float | None = None,
) -> ComparisonResponse:
    """Step D↔E → F/G: Hitung similarity ESCO Skills vs PLO JTK.

    1. Encode semua ESCO skill labels+descriptions via SBERT
    2. Encode semua PLO texts via SBERT
    3. Hitung cosine similarity matrix (D x E)
    4. Untuk tiap ESCO skill: cari PLO dengan skor tertinggi
       - Jika skor >= threshold → masuk F (covered)
       - Jika skor < threshold  → masuk G (gap)

    Args:
        esco_skills: List dict ESCO Skills (dari skill_lookup).
        plo_list: List PLO JTK (dari jtk_repository).
        threshold: Threshold similarity. None = default dari .env.

    Returns:
        ComparisonResponse dengan covered (F) dan gap (G) lists.
    """
    if threshold is None:
        threshold = 0.0  # Tampilkan semua jika tidak diisi

    logger.info(
        "Step D↔E: Comparing %d ESCO skills vs %d PLO (threshold=%.2f) ...",
        len(esco_skills),
        len(plo_list),
        threshold,
    )
    start = time.perf_counter()

    if not esco_skills or not plo_list:
        return ComparisonResponse(
            covered=[],
            gap=[],
            overskill=[],
            threshold_used=threshold,
            total_esco_skills=len(esco_skills),
            total_plo=len(plo_list),
            total_covered=0,
            total_gap=len(esco_skills),
            total_overskill=len(plo_list),
        )

    # Prepare teks untuk encoding
    # Untuk ESCO skills: gabungkan label + description supaya lebih informatif
    skill_texts = [
        f"{s['label']}. {s.get('description', '')}".strip()
        for s in esco_skills
    ]
    plo_texts = [p.plo_text for p in plo_list]

    # Encode
    skill_emb = embeddings.encode(skill_texts)  # shape (D, dim)
    plo_emb = embeddings.encode(plo_texts)  # shape (E, dim)

    # Cosine similarity matrix (D x E)
    sim_matrix = embeddings.cosine_sim(skill_emb, plo_emb)

    # Classify: covered (F) vs gap (G)
    covered: list[CoveredSkill] = []
    gap: list[GapSkill] = []

    for i, skill in enumerate(esco_skills):
        best_plo_idx = int(np.argmax(sim_matrix[i]))
        best_score = float(sim_matrix[i][best_plo_idx])
        best_plo = plo_list[best_plo_idx]

        if best_score >= threshold:
            covered.append(
                CoveredSkill(
                    esco_skill_uri=skill["uri"],
                    esco_skill_label=skill["label"],
                    matched_plo_id=best_plo.plo_id,
                    matched_plo_text=best_plo.plo_text,
                    similarity_score=round(best_score, 4),
                )
            )
        else:
            gap.append(
                GapSkill(
                    esco_skill_uri=skill["uri"],
                    esco_skill_label=skill["label"],
                    best_plo_id=best_plo.plo_id,
                    best_plo_text=best_plo.plo_text,
                    best_similarity_score=round(best_score, 4),
                )
            )

    # Classify: overskill (A-B)
    overskill: list[Overskill] = []
    for j, plo in enumerate(plo_list):
        best_skill_idx = int(np.argmax(sim_matrix[:, j]))
        best_score = float(sim_matrix[best_skill_idx, j])
        best_skill = esco_skills[best_skill_idx]
        
        if best_score < threshold:
            overskill.append(
                Overskill(
                    plo_id=plo.plo_id,
                    plo_text=plo.plo_text,
                    best_esco_skill_uri=best_skill["uri"],
                    best_esco_skill_label=best_skill["label"],
                    best_similarity_score=round(best_score, 4),
                )
            )

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "Step D↔E selesai dalam %.1f ms. Covered (F): %d, Gap (G): %d, Overskill: %d",
        elapsed,
        len(covered),
        len(gap),
        len(overskill),
    )

    return ComparisonResponse(
        covered=covered,
        gap=gap,
        overskill=overskill,
        threshold_used=threshold,
        total_esco_skills=len(esco_skills),
        total_plo=len(plo_list),
        total_covered=len(covered),
        total_gap=len(gap),
        total_overskill=len(overskill),
    )
