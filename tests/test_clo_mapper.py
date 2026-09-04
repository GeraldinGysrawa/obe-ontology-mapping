"""
Tests — CLO Mapper (Step E→K default).

Tests match_plo_to_clo() dengan mock data.
"""

from unittest.mock import patch

import numpy as np
import pytest

from app.models.schemas import CLO, PLO


class TestMatchPloToClo:
    """Test clo_mapper.match_plo_to_clo()."""

    @patch("app.services.clo_mapper.embeddings")
    @patch("app.services.clo_mapper.jtk_repository")
    def test_returns_top_k_clo(self, mock_jtk, mock_emb):
        """Harus return top-K CLO diurutkan by score descending."""
        from app.services.clo_mapper import match_plo_to_clo

        mock_jtk.get_plo_by_id.return_value = PLO(
            plo_id="PLO-01-01",
            plo_text="Menguasai bahasa pemrograman",
            peo_id="PEO-01",
        )
        mock_jtk.get_all_clo.return_value = [
            CLO(clo_id="CLO-JTK-01", clo_text="Belajar pemrograman dasar", mata_kuliah="Dasar Pemrograman", source="jtk"),
            CLO(clo_id="CLO-JTK-02", clo_text="Memahami jaringan komputer", mata_kuliah="Jaringan", source="jtk"),
        ]

        plo_emb = np.array([[0.9, 0.1]])
        clo_emb = np.array([
            [0.85, 0.15],  # high match
            [0.2, 0.8],    # low match
        ])
        mock_emb.encode.side_effect = [plo_emb, clo_emb]
        mock_emb.cosine_sim.return_value = np.dot(
            plo_emb / np.linalg.norm(plo_emb, axis=1, keepdims=True),
            (clo_emb / np.linalg.norm(clo_emb, axis=1, keepdims=True)).T,
        )

        result = match_plo_to_clo("PLO-01-01", top_k=2)

        assert result.plo_id == "PLO-01-01"
        assert len(result.matched_clo) == 2
        assert result.matched_clo[0].score > result.matched_clo[1].score

    @patch("app.services.clo_mapper.jtk_repository")
    def test_plo_not_found(self, mock_jtk):
        """Harus raise ValueError jika PLO tidak ditemukan."""
        from app.services.clo_mapper import match_plo_to_clo

        mock_jtk.get_plo_by_id.return_value = None

        with pytest.raises(ValueError, match="tidak ditemukan"):
            match_plo_to_clo("PLO-99-99")
