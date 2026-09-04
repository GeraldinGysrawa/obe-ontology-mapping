"""
Konfigurasi aplikasi via pydantic-settings.

Semua nilai threshold, path data, nama model SBERT dikonfigurasi via .env
— tidak ada hardcode di kode manapun.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Konfigurasi aplikasi. Semua field dibaca dari environment / .env file."""

    # --- Model SBERT (JANGAN DIGANTI, konsisten seluruh proyek) ---
    sbert_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # --- Path Data ESCO ---
    esco_occupations_path: str = "data/raw/esco/occupations_en.csv"
    esco_skills_path: str = "data/raw/esco/skills_en.csv"
    esco_relations_path: str = "data/raw/esco/occupationSkillRelations_en.csv"

    # --- Path Data JTK ---
    jtk_peo_plo_path: str = "data/raw/jtk/PEO_PLO.csv"
    jtk_clo_jtk_path: str = "data/raw/jtk/CLO-JTK (Jurusan Teknik Komupter).csv"
    jtk_clo_campur_path: str = "data/raw/jtk/CLO-Campur.csv"

    # --- Cache ---
    embeddings_cache_dir: str = "data/processed"

    # --- Threshold ---
    similarity_threshold: float = 0.6

    # --- Top-K defaults ---
    default_top_k: int = 10

    # --- Groq API (untuk NER via LLM) ---
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_api_url: str = "https://api.groq.com/openai/v1/chat/completions"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Cached Settings instance — dipanggil sekali, dipakai di mana-mana."""
    return Settings()
