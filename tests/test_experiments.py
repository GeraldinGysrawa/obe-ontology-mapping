"""
Tests — Experiments (3 metode E→K).

Tests runner output dan export CSV format.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from app.experiments.schemas import MatchResult, MethodName
from app.models.schemas import CLO, PLO


class TestMethodSbertOnly:
    """Test method_sbert_only.run_sbert_only()."""

    @patch("app.experiments.method_sbert_only.embeddings")
    def test_returns_results_with_no_keywords(self, mock_emb):
        """SBERT only: extracted_keywords harus None."""
        from app.experiments.method_sbert_only import run_sbert_only

        clo_list = [
            {"clo_id": "CLO-JTK-01", "clo_text": "Belajar pemrograman"},
        ]

        mock_emb.encode.side_effect = [
            np.array([[1.0, 0.0]]),
            np.array([[0.9, 0.1]]),
        ]
        mock_emb.cosine_sim.return_value = np.array([[0.95]])

        results = run_sbert_only("PLO-01-01", "Programming", clo_list, top_k=1)

        assert len(results) == 1
        assert results[0].method == MethodName.SBERT_ONLY
        assert results[0].extracted_keywords is None
        assert results[0].score_type == "cosine_similarity"


class TestNerFallback:
    """Test NER fallback regex."""

    def test_fallback_extract(self):
        """Fallback regex harus mengekstrak frasa pendek."""
        from app.core.ner import _fallback_extract

        text = "pemrograman web, basis data, jaringan komputer"
        keywords = _fallback_extract(text)

        assert len(keywords) > 0
        assert all(isinstance(k, str) for k in keywords)


class TestExperimentSchemas:
    """Test experiment schemas."""

    def test_match_result_schema(self):
        """MatchResult harus bisa di-instantiate dengan semua field."""
        result = MatchResult(
            plo_id="PLO-01-01",
            clo_id="CLO-JTK-01",
            clo_text="Belajar pemrograman",
            method=MethodName.SBERT_ONLY,
            score=0.85,
            score_type="cosine_similarity",
            extracted_keywords=None,
        )
        assert result.method == MethodName.SBERT_ONLY
        assert result.score == 0.85

    def test_match_result_ner_schema(self):
        """MatchResult NER harus bisa menyimpan keywords."""
        result = MatchResult(
            plo_id="PLO-01-01",
            clo_id="CLO-JTK-01",
            clo_text="Belajar pemrograman",
            method=MethodName.NER_ONLY,
            score=85.5,
            score_type="fuzzy_ratio",
            extracted_keywords=["pemrograman", "web", "database"],
        )
        assert result.method == MethodName.NER_ONLY
        assert result.score_type == "fuzzy_ratio"
        assert len(result.extracted_keywords) == 3
