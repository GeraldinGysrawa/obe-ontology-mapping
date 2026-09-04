"""
Step C→D: Ambil ESCO Skills dari Occupation (relasi langsung ESCO).

Pipeline node: C (ESCO Occupation) --lookup relasi ESCO--> D (ESCO Skills)

Ini adalah lookup langsung dari occupationSkillRelations, BUKAN similarity.
"""

import logging

from app.repositories import esco_repository

logger = logging.getLogger(__name__)


def get_skills_for_occupation(occupation_uri: str) -> list[dict]:
    """Step C→D: Lookup ESCO Skills dari Occupation berdasarkan relasi ESCO.

    Wrapper tipis di atas esco_repository.get_skills_for_occupation().
    Relasi langsung dari occupationSkillRelations_en.csv.

    Args:
        occupation_uri: URI ESCO Occupation (dari hasil Step A→C).

    Returns:
        List dict dengan detail setiap ESCO Skill yang dibutuhkan
        occupation tersebut. Setiap dict berisi:
        - uri: ESCO Skill conceptUri
        - label: preferredLabel
        - skill_type: "skill/competence" atau "knowledge"
        - relation_type: "essential" atau "optional"
        - description: deskripsi skill (bisa kosong)
    """
    logger.info("Step C→D: Lookup skills untuk occupation %s", occupation_uri)

    skills = esco_repository.get_skills_for_occupation(occupation_uri)

    logger.info("Step C→D: Found %d skills", len(skills))

    return skills
