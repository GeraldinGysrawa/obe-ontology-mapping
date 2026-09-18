import logging
import torch
from typing import List, Literal
from app.models.cso_schemas import EscoMatch
from app.core.embeddings import get_model, cosine_sim, load_or_compute_embeddings
from app.repositories import esco_repository
import numpy as np

logger = logging.getLogger(__name__)

class CSOEscoBridge:
    def __init__(self):
        pass

    def pool_embeddings(self, labels: List[str], method: Literal["mean", "concat"] = "mean") -> np.ndarray:
        """
        Create a single embedding representation from a list of expanded CSO labels.
        method='mean': Encode individually and take the mean of the embeddings.
        method='concat': Join the labels into a single string and encode it.
        """
        if not labels:
            raise ValueError("Labels list is empty")
            
        model = get_model()
        
        if method == "mean":
            # Encode individually
            embeddings = model.encode(labels, convert_to_numpy=True)
            # Average across the first dimension (number of labels)
            pooled = np.mean(embeddings, axis=0)
            return pooled
            
        elif method == "concat":
            # Join labels with commas
            joined_text = ", ".join(labels)
            return model.encode([joined_text], convert_to_numpy=True)[0]
            
        else:
            raise ValueError(f"Invalid pooling method: {method}")

    def bridge_to_esco(self, expanded_topic_labels: List[str], top_k: int = 5, pool_method: Literal["mean", "concat"] = "mean") -> List[EscoMatch]:
        """
        Match the expanded CSO topics to ESCO skills.
        """
        if not expanded_topic_labels:
            logger.warning("No expanded topics provided for ESCO bridging.")
            return []
            
        try:
            # Get the pooled embedding for the CSO concepts
            cso_pooled_emb = self.pool_embeddings(expanded_topic_labels, method=pool_method)
            
            # Retrieve ESCO embeddings 
            esco_skills_data = esco_repository.get_all_skill_texts()
            if not esco_skills_data:
                logger.error("ESCO skills data is empty. Cannot bridge to ESCO.")
                return []
                
            texts = [f"{label} {desc}" for _, label, desc in esco_skills_data]
            esco_embeddings = load_or_compute_embeddings(texts, cache_key="esco_skills")
                
            # Calculate cosine similarity
            cosine_scores = cosine_sim(cso_pooled_emb[np.newaxis, :], esco_embeddings)[0]
            
            # Sort scores
            results_with_scores = [
                (esco_skills_data[i], cosine_scores[i].item()) 
                for i in range(len(esco_skills_data))
            ]
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            final_matches = []
            for (uri, label, _), score in results_with_scores[:top_k]:
                final_matches.append(EscoMatch(
                    esco_skill_uri=uri,
                    esco_skill_label=label,
                    score=score
                ))
                
            return final_matches
            
        except Exception as e:
            logger.error(f"Error bridging to ESCO: {str(e)}")
            return []
