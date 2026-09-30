"""
Pydantic schemas untuk modul Computer Science Ontology (CSO).
"""

from typing import List
from pydantic import BaseModel, Field


class TopicMatch(BaseModel):
    """Hasil pencocokan satu keyword ke topik CSO baku."""
    keyword: str | None = Field(default=None, description="Keyword hasil ekstraksi NER")
    topic_uri: str = Field(..., description="URI canonical dari topik CSO")
    topic_name: str = Field(..., description="Nama/Label dari topik CSO")
    score: float = Field(..., description="Skor kemiripan (fuzzy atau cosine)")


class CloToTopicsRequest(BaseModel):
    """Request body untuk POST /cso/clo-to-topics."""
    clo_id: str = Field(
        ...,
        description="ID CLO JTK (mis. 'CLO-JTK-05')",
        examples=["CLO-JTK-05"],
    )


class CloToTopicsResult(BaseModel):
    """Hasil pemetaan CLO ke topik CSO baku."""
    clo_id: str
    mata_kuliah: str = Field("", description="Nama mata kuliah asal CLO")
    clo_text: str = Field("", description="Teks CLO asli")
    extracted_keywords: List[str] = Field(..., description="Keyword hasil NER dari teks CLO")
    matched_topics: List[TopicMatch] = Field(..., description="Topik CSO yang cocok dengan keyword")
