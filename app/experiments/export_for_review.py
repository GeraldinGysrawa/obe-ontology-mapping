"""
Export hasil eksperimen ke CSV untuk evaluasi manual.

Fungsi: export_to_csv(plo_ids, output_path)
- Jalankan run_all_methods untuk semua plo_id
- Tulis ke CSV dengan kolom yang mudah dievaluasi di Excel
- Kolom relevan_manual dan catatan dikosongkan (diisi manual)
"""

import logging
import os

import pandas as pd

from app.experiments.runner import run_all_methods
from app.repositories import jtk_repository

logger = logging.getLogger(__name__)


async def export_to_csv(
    plo_ids: list[str],
    output_path: str,
    top_k: int = 5,
    clo_source: str = "jtk",
) -> str:
    """Ekspor hasil eksperimen ke CSV untuk dinilai manual.

    Jalankan run_all_methods untuk semua plo_id yang diberikan, lalu
    tulis hasilnya ke CSV dengan format yang mudah dievaluasi.

    Args:
        plo_ids: List PLO ID. Kosong = semua PLO.
        output_path: Path file CSV output.
        top_k: Jumlah top-K per metode.
        clo_source: Sumber CLO.

    Returns:
        Path absolut file CSV yang dihasilkan.
    """
    # Jika plo_ids kosong, gunakan semua PLO
    if not plo_ids:
        all_plo = jtk_repository.get_plo_list()
        plo_ids = [p.plo_id for p in all_plo]

    logger.info("Export eksperimen: %d PLO ke %s", len(plo_ids), output_path)

    rows = []

    for plo_id in plo_ids:
        logger.info("Running eksperimen untuk PLO: %s", plo_id)

        try:
            result = await run_all_methods(plo_id, top_k=top_k, clo_source=clo_source)
        except ValueError as e:
            logger.warning("Skip PLO %s: %s", plo_id, e)
            continue

        # Flatten results into rows
        for method_name, timing_result in result.results_per_method.items():
            for match in timing_result.matches:
                rows.append({
                    "plo_id": match.plo_id,
                    "plo_text": result.plo_text,
                    "clo_id": match.clo_id,
                    "clo_text": match.clo_text,
                    "method": match.method.value,
                    "score": match.score,
                    "score_type": match.score_type,
                    "extracted_keywords": (
                        ", ".join(match.extracted_keywords)
                        if match.extracted_keywords
                        else ""
                    ),
                    "relevan_manual": "",  # Dikosongkan, diisi manual
                    "catatan": "",  # Dikosongkan, diisi manual
                })

    # Buat DataFrame dan sort
    df = pd.DataFrame(rows)

    if not df.empty:
        # Sort: plo_id → method → score descending
        method_order = {"sbert_only": 0, "ner_only": 1, "ner_sbert": 2}
        df["_method_order"] = df["method"].map(method_order)
        df = df.sort_values(
            by=["plo_id", "_method_order", "score"],
            ascending=[True, True, False],
        )
        df = df.drop(columns=["_method_order"])

    # Pastikan direktori output ada
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Tulis CSV
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    abs_path = os.path.abspath(output_path)
    logger.info("Export selesai: %s (%d baris)", abs_path, len(df))

    return abs_path
