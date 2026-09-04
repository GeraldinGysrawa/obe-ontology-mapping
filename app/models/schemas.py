"""
Pydantic models (request/response) untuk pipeline utama.

Semua schema API endpoint pipeline utama didefinisikan di sini.
Schema khusus eksperimen ada di app/experiments/schemas.py.
"""

from pydantic import BaseModel, Field


# ── Request Models ─────────────────────────────────────────────────────────


class OccupationMatchRequest(BaseModel):
    """Request body untuk POST /occupation/match (Step A→C)."""

    peo_text: str = Field(
        ...,
        description="Nama PEO / profil lulusan JTK (mis. 'Programmer (Web/Mobile/Desktop)')",
        examples=["Programmer (Web/ Mobile/ Desktop)"],
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Jumlah top-K ESCO Occupation yang dikembalikan",
    )


class ComparisonRequest(BaseModel):
    """Request body untuk POST /comparison/plo-vs-esco (Step D↔E → F/G)."""

    occupation_uri: str = Field(
        ...,
        description="URI ESCO Occupation (dari hasil /occupation/match)",
        examples=["http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1"],
    )
    peo_id: str = Field(
        ...,
        description="ID PEO JTK (mis. 'PEO-01') untuk mengambil PLO terkait",
        examples=["PEO-01"],
    )
    threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Threshold similarity. Jika None, pakai default dari .env",
    )


class CLOMatchRequest(BaseModel):
    """Request body untuk POST /clo/match-from-plo (Step E→K)."""

    plo_id: str = Field(
        ...,
        description="ID PLO JTK (mis. 'PLO-01-01')",
        examples=["PLO-01-01"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Jumlah top-K CLO yang dikembalikan",
    )
    clo_source: str = Field(
        default="jtk",
        description="Sumber CLO: 'jtk' (khusus JTK), 'campur' (termasuk umum), 'all' (gabungan)",
    )


# ── Response Models — Occupation ───────────────────────────────────────────


class OccupationMatch(BaseModel):
    """Satu hasil matching PEO → ESCO Occupation."""

    uri: str = Field(..., description="ESCO Occupation conceptUri")
    label: str = Field(..., description="ESCO Occupation preferredLabel")
    score: float = Field(..., description="Cosine similarity (0-1)")


class OccupationMatchResponse(BaseModel):
    """Response POST /occupation/match."""

    peo_text: str
    matches: list[OccupationMatch]


class SkillDetail(BaseModel):
    """Detail satu ESCO Skill."""

    uri: str = Field(..., description="ESCO Skill conceptUri")
    label: str = Field(..., description="ESCO Skill preferredLabel")
    skill_type: str = Field(..., description="Tipe: 'skill/competence' atau 'knowledge'")
    relation_type: str = Field(..., description="Relasi: 'essential' atau 'optional'")
    description: str | None = Field(default=None, description="Deskripsi skill")


class OccupationSkillsResponse(BaseModel):
    """Response GET /occupation/skills."""

    occupation_uri: str
    occupation_label: str
    total_skills: int
    skills: list[SkillDetail]


# ── Response Models — Comparison (D↔E → F/G) ──────────────────────────────


class CoveredSkill(BaseModel):
    """ESCO Skill yang tercakup oleh PLO JTK (node F)."""

    esco_skill_uri: str
    esco_skill_label: str
    matched_plo_id: str
    matched_plo_text: str
    similarity_score: float


class GapSkill(BaseModel):
    """ESCO Skill yang TIDAK tercakup oleh PLO JTK (node G)."""

    esco_skill_uri: str
    esco_skill_label: str
    best_plo_id: str | None = None
    best_plo_text: str | None = None
    best_similarity_score: float


class ComparisonResponse(BaseModel):
    """Response POST /comparison/plo-vs-esco."""

    covered: list[CoveredSkill]
    gap: list[GapSkill]
    threshold_used: float
    total_esco_skills: int
    total_covered: int
    total_gap: int


# ── Response Models — CLO (E→K) ───────────────────────────────────────────


class MataKuliahInfo(BaseModel):
    """Info singkat mata kuliah (node L)."""

    nama: str


class CLOMatch(BaseModel):
    """Satu hasil matching PLO → CLO."""

    clo_id: str
    clo_text: str
    score: float = Field(..., description="Cosine similarity (0-1)")
    mata_kuliah: MataKuliahInfo


class CLOMatchResponse(BaseModel):
    """Response POST /clo/match-from-plo."""

    plo_id: str
    plo_text: str
    matched_clo: list[CLOMatch]


# ── Internal Data Models (dipakai di repository, bukan response) ──────────


class PEO(BaseModel):
    """Data PEO JTK (node A)."""

    peo_id: str
    peo_text: str  # Nama profil lulusan
    kualifikasi: str
    kompetensi_kerja_raw: str  # Teks mentah "Kompetensi Kerja" (sebelum split)


class PLO(BaseModel):
    """Data PLO JTK (node E) — satu item kompetensi dari PEO."""

    plo_id: str  # Format: PLO-{peo_no}-{item_no}, mis. PLO-01-01
    plo_text: str  # Satu kalimat kompetensi
    peo_id: str  # FK ke PEO


class CLO(BaseModel):
    """Data CLO JTK (node K) — tujuan belajar per mata kuliah."""

    clo_id: str  # Format: CLO-JTK-01 atau CLO-MIX-01
    clo_text: str  # Teks tujuan belajar lengkap
    mata_kuliah: str  # Nama mata kuliah
    source: str  # "jtk" atau "campur"


# ── Response Models — Orchestrator (Pipeline End-to-End) ───────────────────


class PipelineRequest(BaseModel):
    """Request body untuk POST /pipeline/run."""
    peo_name: str = Field(
        ...,
        description="Nama PEO / profil lulusan (mis. 'Programmer')",
        examples=["Programmer"],
    )
    top_k_occupation: int = Field(
        default=3,
        description="Top K occupation yang dicari",
    )


class PipelineMeta(BaseModel):
    """Metadata eksekusi pipeline (untuk laporan skripsi)."""
    clo_method_used: str = "sbert_only"
    execution_time_ms: dict[str, float] = Field(
        description="Durasi (ms) tiap tahap pipeline: 'occupation_match', 'skill_lookup', 'comparison', 'clo_mapping'"
    )


class PipelineResponse(BaseModel):
    """Response lengkap POST /pipeline/run."""
    peo: str
    occupation: OccupationMatch | None = None
    esco_skills: list[SkillDetail] | None = None
    plo_jtk: list[PLO] | None = None
    covered: list[CoveredSkill] | None = None
    gap: list[GapSkill] | None = None
    clo: list[CLOMatchResponse] | None = None
    mata_kuliah: list[str] | None = None
    meta: PipelineMeta
    errors: list[str] = Field(default_factory=list, description="Pesan error jika ada tahap yang gagal/di-skip")
