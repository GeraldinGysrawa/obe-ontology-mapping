"""
Metode 1 — SBERT saja (method_sbert_only).

Eksperimen E→K: Encode kalimat PLO penuh vs kalimat CLO penuh,
hitung cosine similarity, ambil top-K.

Tidak ada ekstraksi keyword — teks digunakan apa adanya.
"""

import logging

import numpy as np

from app.core import embeddings
from app.experiments.schemas import MatchResult, MethodName

logger = logging.getLogger(__name__)


def run_sbert_only(
    plo_id: str,
    plo_text: str,
    clo_list: list[dict],
    top_k: int = 5,
) -> list[MatchResult]:
    """Metode 1: SBERT saja — kalimat PLO penuh vs kalimat CLO penuh.

    Encode plo_text dan semua teks CLO pakai SBERT, hitung cosine
    similarity, urutkan, ambil top-K.

    Args:
        plo_id: ID PLO yang sedang diuji.
        plo_text: Teks PLO lengkap.
        clo_list: List dict CLO, setiap dict punya 'clo_id' dan 'clo_text'.
        top_k: Jumlah top-K hasil.

    Returns:
        List MatchResult — extracted_keywords = None (tidak relevan).
    """
    logger.info("Metode 1 (SBERT saja): PLO=%s, %d CLO", plo_id, len(clo_list))

    if not clo_list:
        return []

    # Encode
    plo_emb = embeddings.encode([plo_text])  # shape (1, D)
    clo_texts = [c["clo_text"] for c in clo_list]
    clo_emb = embeddings.encode(clo_texts)  # shape (N, D)

    # Cosine similarity
    sim_matrix = embeddings.cosine_sim(plo_emb, clo_emb)  # shape (1, N)
    scores = sim_matrix[0]

    # Top-K
    actual_k = min(top_k, len(clo_list))
    top_indices = np.argsort(scores)[::-1][:actual_k]

    results = []
    for idx in top_indices:
        results.append(
            MatchResult(
                plo_id=plo_id,
                clo_id=clo_list[idx]["clo_id"],
                clo_text=clo_list[idx]["clo_text"],
                method=MethodName.SBERT_ONLY,
                score=round(float(scores[idx]), 4),
                score_type="cosine_similarity",
                extracted_keywords=None,
            )
        )

    return results
