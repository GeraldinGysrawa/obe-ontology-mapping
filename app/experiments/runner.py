"""
Runner eksperimen — jalankan 3 metode atas data PLO-CLO yang sama.

Fungsi utama: run_all_methods(plo_id) → ExperimentRunResult
- Ambil teks PLO dari jtk_repository
- Ambil semua CLO JTK
- Jalankan ketiga metode terhadap data yang sama
- Log waktu eksekusi tiap metode (untuk bab hasil skripsi)
"""

import logging
import time

from app.experiments.method_ner_only import run_ner_only
from app.experiments.method_ner_sbert import run_ner_sbert
from app.experiments.method_sbert_only import run_sbert_only
from app.experiments.schemas import (
    ExperimentRunResult,
    MethodName,
    MethodTimingResult,
)
from app.repositories import jtk_repository

logger = logging.getLogger(__name__)


async def run_all_methods(
    plo_id: str,
    top_k: int = 5,
    clo_source: str = "jtk",
) -> ExperimentRunResult:
    """Jalankan ketiga metode eksperimen untuk satu PLO.

    Semua metode dijalankan terhadap data PLO-CLO yang sama supaya
    hasil bisa dibandingkan secara fair. Waktu eksekusi di-log per metode
    untuk perbandingan efisiensi komputasi di bab hasil skripsi.

    Args:
        plo_id: ID PLO yang diuji (mis. "PLO-01-01").
        top_k: Jumlah top-K hasil per metode.
        clo_source: Sumber CLO: "jtk", "campur", atau "all".

    Returns:
        ExperimentRunResult dengan hasil ketiga metode.

    Raises:
        ValueError: Jika plo_id tidak ditemukan.
    """
    # Ambil PLO
    plo = jtk_repository.get_plo_by_id(plo_id)
    if plo is None:
        raise ValueError(f"PLO dengan ID '{plo_id}' tidak ditemukan.")

    plo_text = plo.plo_text

    # Ambil semua CLO
    clo_objects = jtk_repository.get_all_clo(source=clo_source)
    clo_list = [
        {"clo_id": c.clo_id, "clo_text": c.clo_text}
        for c in clo_objects
    ]

    logger.info(
        "Eksperimen: PLO=%s, %d CLO (source=%s), top_k=%d",
        plo_id,
        len(clo_list),
        clo_source,
        top_k,
    )

    results_per_method: dict[MethodName, MethodTimingResult] = {}

    # ── Metode 1: SBERT saja ──────────────────────────────────────────
    start = time.perf_counter()
    matches_sbert = run_sbert_only(plo_id, plo_text, clo_list, top_k)
    duration_sbert = (time.perf_counter() - start) * 1000

    results_per_method[MethodName.SBERT_ONLY] = MethodTimingResult(
        method=MethodName.SBERT_ONLY,
        duration_ms=round(duration_sbert, 1),
        matches=matches_sbert,
    )
    logger.info("Method sbert_only: %.1f ms (%d results)", duration_sbert, len(matches_sbert))

    # ── Metode 2: NER saja ────────────────────────────────────────────
    start = time.perf_counter()
    matches_ner = await run_ner_only(plo_id, plo_text, clo_list, top_k)
    duration_ner = (time.perf_counter() - start) * 1000

    results_per_method[MethodName.NER_ONLY] = MethodTimingResult(
        method=MethodName.NER_ONLY,
        duration_ms=round(duration_ner, 1),
        matches=matches_ner,
    )
    logger.info("Method ner_only: %.1f ms (%d results)", duration_ner, len(matches_ner))

    # ── Metode 3: NER + SBERT ────────────────────────────────────────
    start = time.perf_counter()
    matches_ner_sbert = await run_ner_sbert(plo_id, plo_text, clo_list, top_k)
    duration_ner_sbert = (time.perf_counter() - start) * 1000

    results_per_method[MethodName.NER_SBERT] = MethodTimingResult(
        method=MethodName.NER_SBERT,
        duration_ms=round(duration_ner_sbert, 1),
        matches=matches_ner_sbert,
    )
    logger.info("Method ner_sbert: %.1f ms (%d results)", duration_ner_sbert, len(matches_ner_sbert))

    return ExperimentRunResult(
        plo_id=plo_id,
        plo_text=plo_text,
        results_per_method=results_per_method,
    )
