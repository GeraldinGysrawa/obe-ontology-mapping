"""
Repository JTK — Load & query data kurikulum JTK.

Data files:
- PEO_PLO.csv — 6 PEO (Profil Lulusan) + Kompetensi Kerja (PLO)
  Kolom: No, Bidang Pekerjaan yang dibutuhkan/Profil lulusan,
         Kualifikasi Program Pendidikan, Kompetensi Kerja
  PLO di-split dari teks "Kompetensi Kerja" per nomor item.

- CLO-JTK (Jurusan Teknik Komupter).csv — 33 CLO khusus JTK
- CLO-Campur.csv — 41 CLO (termasuk mata kuliah umum)
  Kolom: No, Nama Mata Kuliah, Tujuan Belajar
  Encoding: latin-1

Relasi dalam pipeline:
- A (PEO) → E (PLO): relasi langsung dari data kurikulum
- K (CLO) → L (Mata Kuliah): embedded dalam file CLO
"""

import logging
import re

import pandas as pd

from app.config import get_settings
from app.models.schemas import CLO, PEO, PLO

logger = logging.getLogger(__name__)

# ── Module-level cache ─────────────────────────────────────────────────────
_peo_list: list[PEO] | None = None
_plo_list: list[PLO] | None = None
_clo_jtk_list: list[CLO] | None = None
_clo_campur_list: list[CLO] | None = None


# ── PEO & PLO ──────────────────────────────────────────────────────────────


def _parse_peo_plo() -> tuple[list[PEO], list[PLO]]:
    """Parse PEO_PLO.csv → list PEO + list PLO.

    Setiap baris CSV = 1 PEO. Kolom "Kompetensi Kerja" dipecah per item
    bernomor menjadi PLO individual.

    ID scheme:
    - PEO: PEO-01, PEO-02, ...
    - PLO: PLO-01-01, PLO-01-02, ... (PEO_no - item_no)
    """
    settings = get_settings()
    path = settings.jtk_peo_plo_path
    logger.info("Loading PEO-PLO dari: %s", path)

    df = pd.read_csv(path)

    # Kolom yang dipakai
    col_no = "No"
    col_profil = "Bidang Pekerjaan yang dibutuhkan/Profil lulusan"
    col_kualifikasi = "Kualifikasi Program Pendidikan"
    col_kompetensi = "Kompetensi Kerja"

    # Drop baris yang No-nya NaN (baris kosong pertama)
    df = df.dropna(subset=[col_no])

    peo_list = []
    plo_list = []

    for _, row in df.iterrows():
        peo_no = int(row[col_no])
        peo_id = f"PEO-{peo_no:02d}"

        peo = PEO(
            peo_id=peo_id,
            peo_text=str(row[col_profil]).strip(),
            kualifikasi=str(row[col_kualifikasi]).strip(),
            kompetensi_kerja_raw=str(row[col_kompetensi]).strip(),
        )
        peo_list.append(peo)

        # Split "Kompetensi Kerja" per item bernomor
        raw_text = str(row[col_kompetensi])
        items = _split_numbered_items(raw_text)

        for item_no, item_text in enumerate(items, start=1):
            plo_id = f"PLO-{peo_no:02d}-{item_no:02d}"
            plo = PLO(
                plo_id=plo_id,
                plo_text=item_text.strip(),
                peo_id=peo_id,
            )
            plo_list.append(plo)

    logger.info("Parsed %d PEO, %d PLO", len(peo_list), len(plo_list))
    return peo_list, plo_list


def _split_numbered_items(text: str) -> list[str]:
    """Split teks multi-item bernomor menjadi list item individual.

    Handles format:
    - "1. text\\n2. text\\n3. text"
    - "1. text 2. text 3. text"

    Args:
        text: Teks mentah kolom "Kompetensi Kerja".

    Returns:
        List teks item individual.
    """
    if not text or text == "nan":
        return []

    # Split berdasar pola "angka." di awal baris atau setelah newline
    # Pattern: satu atau lebih digit, diikuti titik, diikuti spasi
    parts = re.split(r"\n?\s*\d+\.\s+", text)

    # Filter bagian kosong
    items = [p.strip() for p in parts if p.strip()]

    return items


def _ensure_peo_plo_loaded() -> None:
    """Pastikan PEO & PLO sudah di-parse."""
    global _peo_list, _plo_list
    if _peo_list is None or _plo_list is None:
        _peo_list, _plo_list = _parse_peo_plo()


def get_peo_list() -> list[PEO]:
    """Ambil semua PEO JTK (node A).

    Returns:
        List 6 PEO entries.
    """
    _ensure_peo_plo_loaded()
    return _peo_list  # type: ignore


def get_peo_by_id(peo_id: str) -> PEO | None:
    """Ambil satu PEO berdasarkan ID.

    Args:
        peo_id: ID PEO (mis. "PEO-01").

    Returns:
        PEO object atau None.
    """
    for peo in get_peo_list():
        if peo.peo_id == peo_id:
            return peo
    return None


def get_plo_list(peo_id: str | None = None) -> list[PLO]:
    """Ambil semua PLO JTK (node E).

    Relasi A→E: jika peo_id diberikan, filter PLO untuk PEO tersebut.

    Args:
        peo_id: Opsional, filter by PEO ID.

    Returns:
        List PLO entries.
    """
    _ensure_peo_plo_loaded()
    if peo_id:
        return [p for p in _plo_list if p.peo_id == peo_id]  # type: ignore
    return _plo_list  # type: ignore


def get_plo_by_id(plo_id: str) -> PLO | None:
    """Ambil satu PLO berdasarkan ID.

    Args:
        plo_id: ID PLO (mis. "PLO-01-01").

    Returns:
        PLO object atau None.
    """
    for plo in get_plo_list():
        if plo.plo_id == plo_id:
            return plo
    return None


# ── CLO ────────────────────────────────────────────────────────────────────


def _parse_clo_file(path: str, source: str, id_prefix: str) -> list[CLO]:
    """Parse satu file CLO CSV.

    Args:
        path: Path ke file CSV.
        source: Label sumber ("jtk" atau "campur").
        id_prefix: Prefix untuk CLO ID ("CLO-JTK" atau "CLO-MIX").

    Returns:
        List CLO entries.
    """
    logger.info("Loading CLO dari: %s", path)

    df = pd.read_csv(path, encoding="latin-1")

    col_no = "No"
    col_mk = "Nama Mata Kuliah"
    col_clo = "Tujuan Belajar"

    clo_list = []
    for _, row in df.iterrows():
        no = int(row[col_no])
        clo_id = f"{id_prefix}-{no:02d}"

        clo_text = str(row[col_clo]).strip() if pd.notna(row[col_clo]) else ""
        mk_name = str(row[col_mk]).strip() if pd.notna(row[col_mk]) else ""

        clo = CLO(
            clo_id=clo_id,
            clo_text=clo_text,
            mata_kuliah=mk_name,
            source=source,
        )
        clo_list.append(clo)

    logger.info("Parsed %d CLO dari %s", len(clo_list), source)
    return clo_list


def _ensure_clo_loaded() -> None:
    """Pastikan CLO sudah di-parse."""
    global _clo_jtk_list, _clo_campur_list
    settings = get_settings()

    if _clo_jtk_list is None:
        _clo_jtk_list = _parse_clo_file(
            settings.jtk_clo_jtk_path, "jtk", "CLO-JTK"
        )

    if _clo_campur_list is None:
        _clo_campur_list = _parse_clo_file(
            settings.jtk_clo_campur_path, "campur", "CLO-MIX"
        )


def get_all_clo(source: str = "jtk") -> list[CLO]:
    """Ambil semua CLO JTK (node K).

    Args:
        source: "jtk" (khusus JTK, 33 entries), "campur" (termasuk umum, 41 entries),
                atau "all" (gabungan keduanya).

    Returns:
        List CLO entries.
    """
    _ensure_clo_loaded()

    if source == "jtk":
        return _clo_jtk_list  # type: ignore
    elif source == "campur":
        return _clo_campur_list  # type: ignore
    elif source == "all":
        return (_clo_jtk_list or []) + (_clo_campur_list or [])
    else:
        logger.warning("CLO source '%s' tidak dikenal, defaulting ke 'jtk'", source)
        return _clo_jtk_list  # type: ignore


def get_clo_by_id(clo_id: str) -> CLO | None:
    """Ambil satu CLO berdasarkan ID.

    Args:
        clo_id: ID CLO (mis. "CLO-JTK-05" atau "CLO-MIX-01").

    Returns:
        CLO object atau None.
    """
    for clo in get_all_clo(source="all"):
        if clo.clo_id == clo_id:
            return clo
    return None


def get_matakuliah_for_clo(clo_id: str) -> str | None:
    """Ambil nama mata kuliah untuk CLO tertentu (relasi K→L).

    Info mata kuliah embedded dalam data CLO, tidak perlu file terpisah.

    Args:
        clo_id: ID CLO.

    Returns:
        Nama mata kuliah, atau None jika CLO tidak ditemukan.
    """
    clo = get_clo_by_id(clo_id)
    return clo.mata_kuliah if clo else None
