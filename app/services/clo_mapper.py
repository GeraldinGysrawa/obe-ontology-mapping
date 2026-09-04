"""
Step E→K: PLO JTK → CLO JTK (metode SBERT default).

Pipeline node: E (PLO JTK) --SBERT+Cosine--> K (CLO JTK)
               K --relasi langsung--> L (Mata Kuliah JTK)

Metode default SBERT dipakai di luar konteks eksperimen (untuk endpoint
/clo/match-from-plo). Modul eksperimen (3 metode) ada terpisah di
app/experiments/.

# TODO: Mapping langsung D→K (ESCO Skills → CLO) sengaja TIDAK
# diimplementasikan. Gap granularitas antara ESCO Skill (umum) dan
# CLO (teknis/spesifik) membuat mapping langsung ini bermasalah secara
# teori. Lihat diskusi di bab metodologi skripsi. — Future work.
"""

import logging
import time

import numpy as np

from app.core import embeddings
from app.models.schemas import CLOMatch, CLOMatchResponse, MataKuliahInfo
from app.repositories import jtk_repository

logger = logging.getLogger(__name__)


def match_plo_to_clo(
    plo_id: str,
    top_k: int = 5,
    clo_source: str = "jtk",
) -> CLOMatchResponse:
    """Step E→K: Match PLO JTK ke CLO JTK menggunakan SBERT+Cosine.

    1. Ambil teks PLO dari jtk_repository berdasar plo_id
    2. Ambil semua CLO dari jtk_repository (filter by source)
    3. Encode keduanya via SBERT
    4. Hitung cosine similarity, ambil top-K
    5. Untuk tiap CLO match, sertakan info mata kuliah (step K→L)

    Args:
        plo_id: ID PLO JTK (mis. "PLO-01-01").
        top_k: Jumlah top-K CLO yang dikembalikan.
        clo_source: Sumber CLO: "jtk", "campur", atau "all".

    Returns:
        CLOMatchResponse dengan list CLO yang relevan + skor.

    Raises:
        ValueError: Jika plo_id tidak ditemukan.
    """
    logger.info("Step E→K: Matching PLO '%s' ke CLO (source=%s) ...", plo_id, clo_source)
    start = time.perf_counter()

    # Ambil PLO
    plo = jtk_repository.get_plo_by_id(plo_id)
    if plo is None:
        raise ValueError(f"PLO dengan ID '{plo_id}' tidak ditemukan.")

    # Ambil semua CLO
    clo_list = jtk_repository.get_all_clo(source=clo_source)
    if not clo_list:
        return CLOMatchResponse(plo_id=plo_id, plo_text=plo.plo_text, matched_clo=[])

    # Encode PLO text
    plo_emb = embeddings.encode([plo.plo_text])  # shape (1, D)

    # Encode semua CLO texts
    clo_texts = [c.clo_text for c in clo_list]
    clo_emb = embeddings.encode(clo_texts)  # shape (N, D)

    # Cosine similarity
    sim_matrix = embeddings.cosine_sim(plo_emb, clo_emb)  # shape (1, N)
    scores = sim_matrix[0]

    # Top-K
    actual_k = min(top_k, len(clo_list))
    top_indices = np.argsort(scores)[::-1][:actual_k]

    matched_clo = []
    for idx in top_indices:
        clo = clo_list[idx]
        matched_clo.append(
            CLOMatch(
                clo_id=clo.clo_id,
                clo_text=clo.clo_text,
                score=round(float(scores[idx]), 4),
                mata_kuliah=MataKuliahInfo(nama=clo.mata_kuliah),
            )
        )

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "Step E→K selesai dalam %.1f ms. Top-1: %s (%.4f)",
        elapsed,
        matched_clo[0].clo_id if matched_clo else "N/A",
        matched_clo[0].score if matched_clo else 0.0,
    )

    return CLOMatchResponse(
        plo_id=plo_id,
        plo_text=plo.plo_text,
        matched_clo=matched_clo,
    )
