"""
Utilitas pembersihan teks sebelum di-embed oleh SBERT.

Dipanggil sebelum setiap encode() di embeddings.py untuk memastikan
konsistensi input teks ke model.
"""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalisasi teks sebelum di-embed.

    Langkah:
    1. Unicode normalization (NFKD → NFC)
    2. Lowercase
    3. Hapus karakter kontrol
    4. Hapus nomor urut di awal (mis. "1. ", "2. ")
    5. Collapse whitespace berlebih
    6. Strip leading/trailing whitespace

    Args:
        text: Teks mentah dari data kurikulum atau ESCO.

    Returns:
        Teks yang sudah dinormalisasi, siap untuk di-encode.
    """
    if not text or not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFKD", text)
    text = unicodedata.normalize("NFC", text)

    # Lowercase
    text = text.lower()

    # Hapus karakter kontrol (kecuali newline dan tab)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    # Hapus nomor urut di awal baris (mis. "1. ", "2. ", "a. ", "b) ")
    text = re.sub(r"^\s*\d+[\.\)]\s*", "", text)
    text = re.sub(r"\n\s*\d+[\.\)]\s*", "\n", text)

    # Hapus bullet characters
    text = re.sub(r"[•▪▸►‣⁃]", "", text)

    # Collapse whitespace (termasuk newline berlebih)
    text = re.sub(r"\s+", " ", text)

    # Strip
    text = text.strip()

    return text


def clean_text_batch(texts: list[str]) -> list[str]:
    """Bersihkan batch teks sekaligus.

    Args:
        texts: List teks mentah.

    Returns:
        List teks yang sudah dinormalisasi.
    """
    return [clean_text(t) for t in texts]
