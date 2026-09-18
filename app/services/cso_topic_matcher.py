import logging
from rapidfuzz import process, fuzz
from typing import List
from app.models.cso_schemas import TopicMatch
from app.repositories.cso_repository import CSORepository
from app.core.embeddings import get_model, cosine_sim

logger = logging.getLogger(__name__)

class CSOTopicMatcher:
    def __init__(self):
        self.cso_repo = CSORepository()

    def match_keyword_to_topic(self, keyword: str, top_k: int = 3) -> List[TopicMatch]:
        """
        Match a keyword to CSO topics menggunakan SBERT (Semantic Search).
        Tidak menggunakan fuzzy string matching lagi agar mendukung 
        pencocokan silang bahasa (Indonesia -> English).
        """
        all_labels = self.cso_repo.get_all_topic_names()
        if not all_labels:
            logger.warning("No CSO topic labels available for matching.")
            return []

        try:
            # 1. Load SBERT embeddings untuk seluruh topik CSO sekaligus (di-cache)
            from app.core.embeddings import load_or_compute_embeddings, get_model, cosine_sim
            import numpy as np
            
            cso_embs = load_or_compute_embeddings(all_labels, cache_key="cso_topics", clean=False)
            
            # 2. Encode keyword input
            model = get_model()
            keyword_emb = model.encode([keyword], convert_to_numpy=True)
            
            # 3. Hitung cosine similarity dengan seluruh topik CSO
            cosine_scores = cosine_sim(keyword_emb, cso_embs)[0]
            
            # 4. Ambil top_k indeks tertinggi
            top_k_indices = np.argsort(cosine_scores)[-top_k:][::-1]
            
            # 5. Susun hasil
            topic_matches = []
            for idx in top_k_indices:
                label = all_labels[idx]
                score = float(cosine_scores[idx])
                uri = self._get_uri_from_label(label)
                if uri:
                    topic_matches.append(TopicMatch(
                        keyword=keyword,
                        topic_uri=uri,
                        topic_name=label,
                        score=score
                    ))
                    
            return topic_matches
            
        except Exception as e:
            logger.error(f"Error during SBERT matching: {str(e)}")
            return []

    def _get_uri_from_label(self, target_label: str) -> str:
        """Helper to find canonical URI from label. 
        In production, a reverse lookup dict in CSORepository is much more efficient."""
        # For this implementation, we'll do a simple scan.
        for uri, label in self.cso_repo.label_lookup.items():
            if label == target_label:
                return uri
        return ""
