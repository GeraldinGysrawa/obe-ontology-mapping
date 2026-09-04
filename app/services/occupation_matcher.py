"""
Step A→C: PEO JTK → ESCO Occupation via SBERT+Cosine.

Pipeline node: A (PEO JTK) --SBERT+Cosine--> C (ESCO Occupation)

Menerima teks PEO (profil lulusan), encode via SBERT, hitung cosine
similarity terhadap semua ESCO Occupation labels, return top-K.
"""

import logging
import time

import numpy as np

from app.core import embeddings
from app.models.schemas import OccupationMatch
from app.repositories import esco_repository

logger = logging.getLogger(__name__)


def match_peo_to_occupations(
    peo_text: str,
    top_k: int = 10,
) -> list[OccupationMatch]:
    """Step A→C: Match PEO JTK ke ESCO Occupation menggunakan SBERT+Cosine.

    1. Encode peo_text via SBERT
    2. Load/compute cached embeddings untuk semua ESCO occupation labels
    3. Hitung cosine similarity
    4. Return top-K occupation dengan skor

    Args:
        peo_text: Nama/deskripsi PEO/profil lulusan (mis. "Programmer").
        top_k: Jumlah top-K hasil yang dikembalikan.

    Returns:
        List OccupationMatch, diurutkan descending by score.
    """
    logger.info("Step A→C: Matching PEO '%s' ke ESCO Occupations ...", peo_text[:50])
    start = time.perf_counter()

    # Ambil semua ESCO occupation (uri, label)
    occupation_pairs = esco_repository.get_all_occupation_labels()
    uris = [pair[0] for pair in occupation_pairs]
    labels = [pair[1] for pair in occupation_pairs]

    # Encode PEO text
    peo_emb = embeddings.encode([peo_text])  # shape (1, D)

    # Load/compute cached occupation embeddings
    occ_emb = embeddings.load_or_compute_embeddings(
        texts=labels,
        cache_key="esco_occupation_labels",
    )  # shape (N, D)

    # Cosine similarity
    sim_matrix = embeddings.cosine_sim(peo_emb, occ_emb)  # shape (1, N)
    scores = sim_matrix[0]  # shape (N,)

    # Top-K
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        results.append(
            OccupationMatch(
                uri=uris[idx],
                label=labels[idx],
                score=float(scores[idx]),
            )
        )

    elapsed = (time.perf_counter() - start) * 1000
    logger.info(
        "Step A→C selesai dalam %.1f ms. Top-1: %s (%.4f)",
        elapsed,
        results[0].label if results else "N/A",
        results[0].score if results else 0.0,
    )

    return results
