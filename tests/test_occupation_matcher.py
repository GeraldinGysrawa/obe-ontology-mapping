"""
Tests — Occupation Matcher (Step A→C).

Tests menggunakan mock data supaya bisa jalan tanpa data ESCO asli
dan tanpa load model SBERT.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestMatchPeoToOccupations:
    """Test occupation_matcher.match_peo_to_occupations()."""

    @patch("app.services.occupation_matcher.embeddings")
    @patch("app.services.occupation_matcher.esco_repository")
    def test_returns_top_k_results(self, mock_esco, mock_emb):
        """Harus return top-K occupation diurutkan by score descending."""
        from app.services.occupation_matcher import match_peo_to_occupations

        # Mock data: 3 occupations
        mock_esco.get_all_occupation_labels.return_value = [
            ("uri_1", "Software Developer"),
            ("uri_2", "Data Analyst"),
            ("uri_3", "Network Engineer"),
        ]

        # Mock embeddings
        peo_emb = np.array([[1.0, 0.0, 0.0]])
        occ_emb = np.array([
            [0.9, 0.1, 0.0],   # high similarity
            [0.1, 0.9, 0.0],   # low similarity
            [0.5, 0.5, 0.0],   # medium similarity
        ])
        mock_emb.encode.return_value = peo_emb
        mock_emb.load_or_compute_embeddings.return_value = occ_emb
        mock_emb.cosine_sim.return_value = np.dot(
            peo_emb / np.linalg.norm(peo_emb, axis=1, keepdims=True),
            (occ_emb / np.linalg.norm(occ_emb, axis=1, keepdims=True)).T,
        )

        results = match_peo_to_occupations("Programmer", top_k=2)

        assert len(results) == 2
        # First result should have highest score
        assert results[0].score > results[1].score
        assert results[0].uri == "uri_1"

    @patch("app.services.occupation_matcher.embeddings")
    @patch("app.services.occupation_matcher.esco_repository")
    def test_empty_occupations(self, mock_esco, mock_emb):
        """Harus handle case tanpa occupation."""
        from app.services.occupation_matcher import match_peo_to_occupations

        mock_esco.get_all_occupation_labels.return_value = []
        mock_emb.encode.return_value = np.array([[1.0, 0.0]])
        mock_emb.load_or_compute_embeddings.return_value = np.empty((0, 2))
        mock_emb.cosine_sim.return_value = np.empty((1, 0))

        results = match_peo_to_occupations("Test", top_k=5)
        assert len(results) == 0
