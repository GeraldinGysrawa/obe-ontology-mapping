"""
Metode 3 — NER + SBERT (method_ner_sbert).

Eksperimen E→K: NER ekstrak keyword dari setiap CLO, gabungkan keyword
jadi satu string (join koma), encode via SBERT, hitung cosine similarity
terhadap embedding PLO.

Menggabungkan kekuatan NER (ekstraksi konsep kunci) dengan SBERT
(semantic similarity) untuk matching yang lebih terarah.
"""

import logging

import numpy as np

from app.core import embeddings
from app.core.ner import extract_skills_ner
from app.experiments.schemas import MatchResult, MethodName

logger = logging.getLogger(__name__)


async def run_ner_sbert(
    plo_id: str,
    plo_text: str,
    clo_list: list[dict],
    top_k: int = 5,
) -> list[MatchResult]:
    """Metode 3: NER + SBERT — ekstrak keyword CLO, encode SBERT, cosine ke PLO.

    Untuk setiap CLO:
    1. Ekstrak keyword via NER (sama seperti Metode 2)
    2. Gabungkan keyword jadi satu string pendek (join dengan koma)
    3. Encode string keyword gabungan via SBERT
    4. Hitung cosine similarity terhadap embedding plo_text

    Args:
        plo_id: ID PLO yang sedang diuji.
        plo_text: Teks PLO lengkap.
        clo_list: List dict CLO, setiap dict punya 'clo_id' dan 'clo_text'.
        top_k: Jumlah top-K hasil.

    Returns:
        List MatchResult — score dalam skala 0-1 (cosine_similarity),
        extracted_keywords berisi keyword yang diekstrak.
    """
    logger.info("Metode 3 (NER+SBERT): PLO=%s, %d CLO", plo_id, len(clo_list))

    if not clo_list:
        return []

    # Encode PLO text
    plo_emb = embeddings.encode([plo_text])  # shape (1, D)

    # Ekstrak keyword dari setiap CLO dan gabungkan jadi string
    keyword_strings = []
    all_keywords = []

    for clo in clo_list:
        keywords = await extract_skills_ner(clo["clo_text"])
        all_keywords.append(keywords)

        if keywords:
            keyword_str = ", ".join(keywords)
        else:
            # Fallback: gunakan teks CLO asli jika NER gagal
            keyword_str = clo["clo_text"]

        keyword_strings.append(keyword_str)

    # Encode keyword strings via SBERT
    kw_emb = embeddings.encode(keyword_strings)  # shape (N, D)

    # Cosine similarity
    sim_matrix = embeddings.cosine_sim(plo_emb, kw_emb)  # shape (1, N)
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
                method=MethodName.NER_SBERT,
                score=round(float(scores[idx]), 4),
                score_type="cosine_similarity",
                extracted_keywords=all_keywords[idx],
            )
        )

    return results
