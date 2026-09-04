"""
Repository ESCO — Load & query data ESCO dari data/raw/esco/.

Data files:
- occupations_en.csv (3043 rows)
- skills_en.csv (13960 rows)
- occupationSkillRelations_en.csv (126051 rows)

Semua DataFrame di-cache di module-level setelah load pertama (lazy loading).
"""

import logging

import pandas as pd

from app.config import get_settings

logger = logging.getLogger(__name__)

# ── Module-level cache ─────────────────────────────────────────────────────
_occupations_df: pd.DataFrame | None = None
_skills_df: pd.DataFrame | None = None
_relations_df: pd.DataFrame | None = None


# ── Load functions ─────────────────────────────────────────────────────────


def load_occupations() -> pd.DataFrame:
    """Load dan cache occupations_en.csv.

    Returns:
        DataFrame dengan kolom: conceptUri, preferredLabel, altLabels, description, dll.
    """
    global _occupations_df
    if _occupations_df is not None:
        return _occupations_df

    settings = get_settings()
    path = settings.esco_occupations_path
    logger.info("Loading ESCO occupations dari: %s", path)

    _occupations_df = pd.read_csv(path)
    logger.info("ESCO occupations loaded: %d rows", len(_occupations_df))

    return _occupations_df


def load_skills() -> pd.DataFrame:
    """Load dan cache skills_en.csv.

    Returns:
        DataFrame dengan kolom: conceptUri, preferredLabel, description, skillType, dll.
    """
    global _skills_df
    if _skills_df is not None:
        return _skills_df

    settings = get_settings()
    path = settings.esco_skills_path
    logger.info("Loading ESCO skills dari: %s", path)

    _skills_df = pd.read_csv(path)
    logger.info("ESCO skills loaded: %d rows", len(_skills_df))

    return _skills_df


def load_relations() -> pd.DataFrame:
    """Load dan cache occupationSkillRelations_en.csv.

    Returns:
        DataFrame dengan kolom: occupationUri, skillUri, relationType,
        skillType, occupationLabel, skillLabel.
    """
    global _relations_df
    if _relations_df is not None:
        return _relations_df

    settings = get_settings()
    path = settings.esco_relations_path
    logger.info("Loading ESCO occupation-skill relations dari: %s", path)

    _relations_df = pd.read_csv(path)
    logger.info("ESCO relations loaded: %d rows", len(_relations_df))

    return _relations_df


# ── Query functions ────────────────────────────────────────────────────────


def get_occupation_by_uri(uri: str) -> dict | None:
    """Ambil satu occupation berdasarkan conceptUri.

    Args:
        uri: ESCO Occupation URI.

    Returns:
        Dict dengan data occupation, atau None jika tidak ditemukan.
    """
    df = load_occupations()
    match = df[df["conceptUri"] == uri]

    if match.empty:
        return None

    row = match.iloc[0]
    return {
        "uri": row["conceptUri"],
        "label": row["preferredLabel"],
        "description": row.get("description", ""),
        "alt_labels": row.get("altLabels", ""),
    }


def get_skills_for_occupation(occupation_uri: str) -> list[dict]:
    """Step C→D: Ambil semua ESCO Skills yang dibutuhkan occupation.

    Lookup langsung dari occupationSkillRelations, BUKAN similarity.
    Join dengan skills_en.csv untuk mendapatkan description.

    Args:
        occupation_uri: URI ESCO Occupation.

    Returns:
        List dict dengan detail setiap skill.
    """
    relations_df = load_relations()
    skills_df = load_skills()

    # Filter relasi untuk occupation ini
    rel_filtered = relations_df[relations_df["occupationUri"] == occupation_uri]

    if rel_filtered.empty:
        logger.warning("Tidak ada skill untuk occupation: %s", occupation_uri)
        return []

    # Join dengan skills untuk dapat description
    merged = rel_filtered.merge(
        skills_df[["conceptUri", "description"]],
        left_on="skillUri",
        right_on="conceptUri",
        how="left",
        suffixes=("", "_skill"),
    )

    skills = []
    for _, row in merged.iterrows():
        skills.append({
            "uri": row["skillUri"],
            "label": row["skillLabel"],
            "skill_type": row["skillType"],
            "relation_type": row["relationType"],
            "description": row.get("description", "") if pd.notna(row.get("description", "")) else "",
        })

    logger.info(
        "Found %d skills untuk occupation %s",
        len(skills),
        occupation_uri,
    )

    return skills


def get_all_occupation_labels() -> list[tuple[str, str]]:
    """Ambil semua (uri, preferredLabel) dari ESCO Occupations.

    Digunakan untuk batch encoding saat occupation matching (Step A→C).

    Returns:
        List of tuples (conceptUri, preferredLabel).
    """
    df = load_occupations()
    return list(zip(df["conceptUri"].tolist(), df["preferredLabel"].tolist()))


def get_all_skill_texts() -> list[tuple[str, str, str]]:
    """Ambil semua (uri, label, description) dari ESCO Skills.

    Digunakan untuk batch encoding saat skill comparison (Step D↔E).

    Returns:
        List of tuples (conceptUri, preferredLabel, description).
    """
    df = load_skills()
    descriptions = df["description"].fillna("").tolist()
    return list(zip(
        df["conceptUri"].tolist(),
        df["preferredLabel"].tolist(),
        descriptions,
    ))
