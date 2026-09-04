"""
Wrapper SBERT: load model SEKALI (singleton), encode(), cosine_sim().

Model yang dipakai DI SELURUH PROYEK tanpa kecuali:
  sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Model di-load SEKALI saat startup (via load_model()), disimpan di module-level
variable, dipakai bersama oleh pipeline utama maupun modul eksperimen.

Cache mekanisme:
  load_or_compute_embeddings() — cek file .npy di data/processed/,
  jika ada langsung load, jika tidak compute lalu simpan.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.utils.text_cleaning import clean_text_batch

logger = logging.getLogger(__name__)

# ── Singleton model ────────────────────────────────────────────────────────
_model: SentenceTransformer | None = None


def load_model() -> None:
    """Load model SBERT SEKALI saat aplikasi startup.

    Dipanggil dari lifespan handler di main.py.
    Model disimpan di module-level variable supaya tidak di-load ulang.
    """
    global _model
    if _model is not None:
        logger.info("SBERT model sudah ter-load, skip.")
        return

    settings = get_settings()
    model_name = settings.sbert_model_name

    logger.info("Loading SBERT model: %s ...", model_name)
    start = time.perf_counter()
    _model = SentenceTransformer(model_name)
    elapsed = (time.perf_counter() - start) * 1000
    logger.info("SBERT model loaded dalam %.1f ms", elapsed)


def get_model() -> SentenceTransformer:
    """Ambil instance model SBERT.

    Raises:
        RuntimeError: Jika model belum di-load via load_model().
    """
    if _model is None:
        raise RuntimeError(
            "SBERT model belum di-load. Panggil load_model() saat startup."
        )
    return _model


# ── Encode ─────────────────────────────────────────────────────────────────

def encode(texts: list[str], clean: bool = True) -> np.ndarray:
    """Encode batch teks menjadi embeddings.

    Args:
        texts: List teks yang akan di-encode.
        clean: Jika True, teks dinormalisasi dulu via text_cleaning.

    Returns:
        np.ndarray shape (len(texts), embedding_dim).
    """
    model = get_model()

    if clean:
        texts = clean_text_batch(texts)

    logger.debug("Encoding %d teks ...", len(texts))
    start = time.perf_counter()
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    elapsed = (time.perf_counter() - start) * 1000
    logger.debug("Encoding selesai dalam %.1f ms", elapsed)

    return embeddings


# ── Cosine similarity ──────────────────────────────────────────────────────

def cosine_sim(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """Hitung cosine similarity matrix antara dua set embeddings.

    Args:
        emb_a: np.ndarray shape (M, D)
        emb_b: np.ndarray shape (N, D)

    Returns:
        np.ndarray shape (M, N) — similarity matrix.
    """
    # Normalize
    norm_a = emb_a / np.linalg.norm(emb_a, axis=1, keepdims=True)
    norm_b = emb_b / np.linalg.norm(emb_b, axis=1, keepdims=True)

    return np.dot(norm_a, norm_b.T)


# ── Cache mechanism ───────────────────────────────────────────────────────

def load_or_compute_embeddings(
    texts: list[str],
    cache_key: str,
    cache_dir: str | None = None,
    clean: bool = True,
) -> np.ndarray:
    """Load embeddings dari cache .npy, atau compute + simpan jika belum ada.

    Mekanisme load-if-exists-else-compute untuk menghindari re-encode
    data besar (mis. 13.960 ESCO Skills) setiap request.

    Args:
        texts: List teks yang akan di-encode.
        cache_key: Nama unik untuk file cache (tanpa ekstensi).
        cache_dir: Direktori cache. Default dari settings.
        clean: Jika True, teks dinormalisasi sebelum encode.

    Returns:
        np.ndarray shape (len(texts), embedding_dim).
    """
    if cache_dir is None:
        cache_dir = get_settings().embeddings_cache_dir

    cache_path = Path(cache_dir) / f"{cache_key}.npy"

    # Load from cache if exists and size matches
    if cache_path.exists():
        logger.info("Loading cached embeddings: %s", cache_path)
        cached = np.load(str(cache_path))
        if cached.shape[0] == len(texts):
            logger.info(
                "Cache hit: %s (%d embeddings)", cache_key, cached.shape[0]
            )
            return cached
        else:
            logger.warning(
                "Cache size mismatch (%d vs %d), re-computing ...",
                cached.shape[0],
                len(texts),
            )

    # Compute embeddings
    logger.info("Computing embeddings untuk '%s' (%d teks) ...", cache_key, len(texts))
    embeddings = encode(texts, clean=clean)

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)
    np.save(str(cache_path), embeddings)
    logger.info("Embeddings saved to cache: %s", cache_path)

    return embeddings
