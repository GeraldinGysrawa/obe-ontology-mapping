from pydantic import BaseModel, Field
from typing import List, Optional


class TopicMatch(BaseModel):
    keyword: str | None = Field(default=None, description="Keyword yang diekstrak dan dicocokkan")
    topic_uri: str = Field(..., description="Canonical URI of the matched CSO topic")
    topic_name: str = Field(..., description="Label of the matched CSO topic")
    score: float = Field(..., description="Similarity score (fuzzy or cosine)")


class EscoMatch(BaseModel):
    esco_skill_uri: str = Field(..., description="URI of the ESCO Skill")
    esco_skill_label: str = Field(..., description="Label of the ESCO Skill")
    score: float = Field(..., description="Cosine similarity score")


class CloToEscoViaCsoResult(BaseModel):
    """Legacy schema — dipertahankan untuk backward compatibility."""
    clo_id: str
    extracted_keywords: List[str]
    matched_topics: List[TopicMatch]
    expanded_topics: List[str] = Field(default_factory=list, description="List of labels for the expanded topics")
    final_esco_matches: List[EscoMatch]


# ── Endpoint 1: CLO → CSO Topics saja (tanpa ESCO) ────────────────────────

class CloToTopicsResult(BaseModel):
    """Hasil pemetaan CLO ke topik CSO baku — TANPA perbandingan ke ESCO."""
    clo_id: str
    mata_kuliah: str = Field("", description="Nama mata kuliah asal CLO")
    clo_text: str = Field("", description="Teks CLO asli")
    extracted_keywords: List[str] = Field(..., description="Keyword hasil NER dari teks CLO")
    matched_topics: List[TopicMatch] = Field(..., description="Topik CSO yang cocok dengan keyword")


# ── Endpoint 2: CLO → CSO → ESCO Gap Analysis ─────────────────────────────

class CsoEscoCovered(BaseModel):
    """Satu topik CSO yang TERCAKUP oleh ESCO Skill (skor >= threshold)."""
    cso_topic: str
    best_esco_skill_uri: str
    best_esco_skill_label: str
    similarity_score: float


class CsoEscoGap(BaseModel):
    """Satu topik CSO yang BELUM TERCAKUP oleh ESCO Skill (skor < threshold)."""
    cso_topic: str
    best_esco_skill_uri: str
    best_esco_skill_label: str
    best_similarity_score: float


class CloEscoGapResult(BaseModel):
    """Hasil gap analysis level CLO: topik CSO vs ESCO Skills."""
    clo_id: str
    mata_kuliah: str = ""
    clo_text: str = ""
    extracted_keywords: List[str] = Field(default_factory=list)
    matched_topics: List[TopicMatch] = Field(default_factory=list)
    threshold_used: float
    total_cso_topics: int
    total_covered: int
    total_gap: int
    covered: List[CsoEscoCovered] = Field(default_factory=list)
    gap: List[CsoEscoGap] = Field(default_factory=list)

