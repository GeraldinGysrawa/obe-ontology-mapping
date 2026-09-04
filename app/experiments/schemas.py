"""
Pydantic model khusus hasil eksperimen perbandingan 3 metode E→K.

Schema ini terpisah dari app/models/schemas.py karena hanya dipakai
di konteks eksperimen, bukan pipeline utama.
"""

from enum import Enum

from pydantic import BaseModel, Field


class MethodName(str, Enum):
    """Enum nama metode eksperimen."""

    SBERT_ONLY = "sbert_only"
    NER_ONLY = "ner_only"
    NER_SBERT = "ner_sbert"


class MatchResult(BaseModel):
    """Satu hasil matching PLO → CLO dari satu metode."""

    plo_id: str
    clo_id: str
    clo_text: str
    method: MethodName
    score: float | None = Field(
        default=None,
        description="Skor matching. Skala 0-1 untuk cosine similarity, 0-100 untuk fuzzy ratio.",
    )
    score_type: str = Field(
        ...,
        description="Tipe skor: 'cosine_similarity' (0-1) atau 'fuzzy_ratio' (0-100)",
    )
    extracted_keywords: list[str] | None = Field(
        default=None,
        description="Keyword yang diekstrak NER (None untuk metode SBERT saja)",
    )


class MethodTimingResult(BaseModel):
    """Hasil satu metode eksperimen, termasuk timing."""

    method: MethodName
    duration_ms: float = Field(..., description="Durasi eksekusi dalam milidetik")
    matches: list[MatchResult]


class ExperimentRunResult(BaseModel):
    """Hasil lengkap eksperimen untuk satu PLO — gabungan 3 metode."""

    plo_id: str
    plo_text: str
    results_per_method: dict[MethodName, MethodTimingResult]


class ExperimentExportRequest(BaseModel):
    """Request body untuk POST /experiments/export."""

    plo_ids: list[str] = Field(
        default_factory=list,
        description="List PLO ID. Kosong = semua PLO.",
    )
