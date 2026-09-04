"""
Tests — Skill Comparator (Step D↔E → F/G).

Tests threshold logic: covered vs gap classification.
"""

from unittest.mock import patch

import numpy as np
import pytest

from app.models.schemas import PLO


class TestCompareSkillsVsPlo:
    """Test skill_comparator.compare_skills_vs_plo()."""

    @patch("app.services.skill_comparator.embeddings")
    def test_covered_and_gap_split(self, mock_emb):
        """Skills di atas threshold → covered, di bawah → gap."""
        from app.services.skill_comparator import compare_skills_vs_plo

        esco_skills = [
            {"uri": "skill_1", "label": "Python programming", "description": ""},
            {"uri": "skill_2", "label": "Quantum physics", "description": ""},
        ]

        plo_list = [
            PLO(plo_id="PLO-01-01", plo_text="Menguasai bahasa pemrograman Python", peo_id="PEO-01"),
        ]

        # Mock: skill_1 has high similarity (0.8), skill_2 has low (0.3)
        mock_emb.encode.side_effect = [
            np.array([[0.9, 0.1], [0.1, 0.9]]),  # skill embeddings
            np.array([[0.85, 0.15]]),  # PLO embedding
        ]
        mock_emb.cosine_sim.return_value = np.array([
            [0.8],   # skill_1 vs PLO
            [0.3],   # skill_2 vs PLO
        ])

        result = compare_skills_vs_plo(esco_skills, plo_list, threshold=0.6)

        assert result.total_covered == 1
        assert result.total_gap == 1
        assert result.covered[0].esco_skill_uri == "skill_1"
        assert result.gap[0].esco_skill_uri == "skill_2"
        assert result.threshold_used == 0.6

    @patch("app.services.skill_comparator.embeddings")
    def test_all_covered(self, mock_emb):
        """Semua skills di atas threshold → semua covered."""
        from app.services.skill_comparator import compare_skills_vs_plo

        esco_skills = [
            {"uri": "skill_1", "label": "Programming", "description": ""},
        ]
        plo_list = [
            PLO(plo_id="PLO-01-01", plo_text="Programming skill", peo_id="PEO-01"),
        ]

        mock_emb.encode.side_effect = [
            np.array([[1.0, 0.0]]),
            np.array([[0.95, 0.05]]),
        ]
        mock_emb.cosine_sim.return_value = np.array([[0.95]])

        result = compare_skills_vs_plo(esco_skills, plo_list, threshold=0.5)

        assert result.total_covered == 1
        assert result.total_gap == 0

    def test_empty_inputs(self):
        """Harus handle input kosong tanpa error."""
        from app.services.skill_comparator import compare_skills_vs_plo

        result = compare_skills_vs_plo([], [], threshold=0.6)
        assert result.total_covered == 0
        assert result.total_gap == 0
