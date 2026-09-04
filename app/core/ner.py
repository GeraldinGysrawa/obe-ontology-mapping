"""
NER (Named Entity Recognition) — Ekstraksi keyword/skill dari teks CLO.

Implementasi menggunakan Groq LLM via API dengan few-shot prompting.
Digunakan di modul eksperimen (Metode 2: NER saja, Metode 3: NER+SBERT).

Fallback ke regex sederhana jika Groq API gagal, supaya pipeline
tidak crash saat testing offline.
"""

import json
import logging
import re
import time

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Few-shot prompt template ───────────────────────────────────────────────

_SYSTEM_PROMPT = """Kamu adalah asisten ekstraksi keyword dari teks tujuan belajar (CLO) mata kuliah.
Tugasmu: dari teks CLO yang diberikan, ekstrak keyword/skill teknis yang penting.

Aturan:
- Ekstrak 3-10 keyword utama (skill, konsep, kemampuan teknis)
- Keyword harus singkat (1-4 kata)
- Fokus pada skill teknis, bukan kata penghubung
- Kembalikan sebagai JSON array of strings
- HANYA kembalikan JSON array, tanpa penjelasan tambahan

Contoh:

Input: "Mahasiswa mampu menyusun dan menstrukturkan logika berpikir secara sistematis untuk memahami konsep berpikir logis dan komputasi sebagai modal dasar dalam memecahkan masalah"
Output: ["logika berpikir", "berpikir logis", "komputasi", "pemecahan masalah", "berpikir sistematis"]

Input: "Memahami konsep dasar sistem komputer, perangkat keras, perangkat lunak, pemikiran komputasi, dan literasi digital"
Output: ["sistem komputer", "perangkat keras", "perangkat lunak", "pemikiran komputasi", "literasi digital"]

Input: "Mampu mengembangkan aplikasi berbasis web menggunakan framework modern dan menerapkan prinsip UI/UX"
Output: ["pengembangan aplikasi web", "framework modern", "UI/UX", "aplikasi berbasis web"]"""


async def extract_skills_ner(text: str) -> list[str]:
    """Ekstrak keyword/skill dari teks CLO menggunakan Groq LLM.

    Digunakan di modul eksperimen (Metode 2 dan 3) untuk mengekstrak
    keyword teknis dari teks CLO sebelum matching ke PLO.

    Args:
        text: Teks CLO yang akan diekstrak keyword-nya.

    Returns:
        List keyword/skill yang diekstrak.
        Jika API gagal, fallback ke regex sederhana.
    """
    settings = get_settings()

    if not settings.groq_api_key or settings.groq_api_key.startswith("gsk_XXXX"):
        logger.warning("GROQ_API_KEY belum dikonfigurasi, menggunakan fallback regex.")
        return _fallback_extract(text)

    try:
        return await _call_groq_api(text, settings)
    except Exception as e:
        logger.error("Groq API gagal: %s — menggunakan fallback regex.", str(e))
        return _fallback_extract(text)


async def _call_groq_api(text: str, settings) -> list[str]:
    """Panggil Groq API dengan few-shot prompt untuk ekstraksi keyword.

    Args:
        text: Teks CLO.
        settings: App settings instance.

    Returns:
        List keyword dari response LLM.
    """
    start = time.perf_counter()

    payload = {
        "model": settings.groq_model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0.1,  # rendah supaya deterministik
        "max_tokens": 300,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.groq_api_key}",
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            settings.groq_api_url,
            json=payload,
            headers=headers,
        )
        response.raise_for_status()

    elapsed = (time.perf_counter() - start) * 1000
    data = response.json()

    # Parse response
    content = data["choices"][0]["message"]["content"].strip()

    # Coba parse sebagai JSON array
    keywords = _parse_json_array(content)

    logger.info(
        "NER via Groq: %d keywords diekstrak dalam %.1f ms",
        len(keywords),
        elapsed,
    )

    return keywords


def _parse_json_array(content: str) -> list[str]:
    """Parse response LLM menjadi list[str].

    Handles beberapa format output yang mungkin:
    - Pure JSON array: ["a", "b", "c"]
    - JSON array dalam code block: ```json\n["a"]\n```

    Args:
        content: Raw response text dari LLM.

    Returns:
        List keyword.
    """
    # Hapus code block markers jika ada
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)
    content = content.strip()

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(k).strip() for k in parsed if str(k).strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: coba ekstrak array dari dalam teks
    match = re.search(r"\[.*?\]", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                return [str(k).strip() for k in parsed if str(k).strip()]
        except json.JSONDecodeError:
            pass

    logger.warning("Gagal parse response LLM sebagai JSON array: %s", content[:100])
    return _fallback_extract(content)


def _fallback_extract(text: str) -> list[str]:
    """Fallback regex sederhana untuk ekstraksi keyword.

    Digunakan jika Groq API gagal atau belum dikonfigurasi.
    Heuristik: split kalimat, ambil frasa pendek yang mengandung
    kata kunci teknis.

    Args:
        text: Teks CLO.

    Returns:
        List keyword (best-effort).

    # TODO: Ganti dengan model NER kustom ("NER Gege") untuk
    # akurasi yang lebih baik. Regex ini hanya placeholder.
    """
    if not text:
        return []

    # Split berdasar koma, titik koma, newline
    fragments = re.split(r"[,;\n]", text)

    keywords = []
    for frag in fragments:
        frag = frag.strip()
        # Skip fragmen terlalu pendek atau terlalu panjang
        if len(frag) < 3 or len(frag) > 80:
            continue
        # Hapus nomor urut di awal
        frag = re.sub(r"^\d+[\.\)]\s*", "", frag).strip()
        if frag and len(frag.split()) <= 6:
            keywords.append(frag.lower())

    # Deduplicate
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    return unique[:10]  # Limit ke 10 keyword
