"""
Metode 2 — NER saja (method_ner_only).

Eksperimen E→K: NER ekstrak keyword dari setiap CLO, fuzzy match
tiap keyword ke teks PLO menggunakan rapidfuzz.fuzz.partial_ratio().
Skor akhir CLO = skor fuzzy tertinggi dari semua keyword-nya.

Skala skor: 0-100 (fuzzy ratio), BUKAN 0-1 seperti cosine similarity.
"""

import logging

from rapidfuzz import fuzz

from app.core.ner import extract_skills_ner
from app.experiments.schemas import MatchResult, MethodName

logger = logging.getLogger(__name__)


async def run_ner_only(
    plo_id: str,
    plo_text: str,
    clo_list: list[dict],
    top_k: int = 5,
) -> list[MatchResult]:
    """Metode 2: NER saja — ekstrak keyword CLO, fuzzy match ke PLO.

    Untuk setiap CLO:
    1. Panggil extract_skills_ner(clo_text) → list keyword
    2. Untuk tiap keyword, hitung rapidfuzz.fuzz.partial_ratio() vs plo_text
    3. Skor akhir CLO = skor fuzzy tertinggi dari semua keyword-nya

    Args:
        plo_id: ID PLO yang sedang diuji.
        plo_text: Teks PLO lengkap.
        clo_list: List dict CLO, setiap dict punya 'clo_id' dan 'clo_text'.
        top_k: Jumlah top-K hasil.

    Returns:
        List MatchResult — score dalam skala 0-100 (fuzzy_ratio),
        extracted_keywords berisi keyword yang diekstrak.
    """
    logger.info("Metode 2 (NER saja): PLO=%s, %d CLO", plo_id, len(clo_list))

    if not clo_list:
        return []

    plo_lower = plo_text.lower()

    scored_results = []

    for clo in clo_list:
        # Ekstrak keyword dari CLO via NER
        keywords = await extract_skills_ner(clo["clo_text"])

        if not keywords:
            # Tidak ada keyword → skor 0
            scored_results.append({
                "clo_id": clo["clo_id"],
                "clo_text": clo["clo_text"],
                "score": 0.0,
                "keywords": [],
            })
            continue

        # Fuzzy match tiap keyword ke plo_text
        max_score = 0.0
        for kw in keywords:
            ratio = fuzz.partial_ratio(kw.lower(), plo_lower)
            if ratio > max_score:
                max_score = ratio

        scored_results.append({
            "clo_id": clo["clo_id"],
            "clo_text": clo["clo_text"],
            "score": max_score,
            "keywords": keywords,
        })

    # Sort descending by score, ambil top-K
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    top_results = scored_results[:top_k]

    results = []
    for item in top_results:
        results.append(
            MatchResult(
                plo_id=plo_id,
                clo_id=item["clo_id"],
                clo_text=item["clo_text"],
                method=MethodName.NER_ONLY,
                score=round(item["score"], 2),
                score_type="fuzzy_ratio",
                extracted_keywords=item["keywords"],
            )
        )

    return results
