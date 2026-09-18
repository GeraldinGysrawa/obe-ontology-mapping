import logging
from typing import List, Literal
from app.repositories.cso_repository import CSORepository

logger = logging.getLogger(__name__)

class CSOGraphExpander:
    def __init__(self):
        self.cso_repo = CSORepository()

    def expand_topic(self, topic_uri: str, mode: Literal["hierarchy", "contribution"] = "hierarchy", hop: int = 1, direction: Literal["up", "down", "both"] = "up") -> List[str]:
        """
        Expand a topic to its neighbors in the CSO graph.
        Returns a list of topic labels (not URIs) of the expanded topics.
        """
        if not self.cso_repo._is_loaded:
            logger.warning("CSORepository not loaded!")
            return []

        # Ensure we start with canonical URI
        canon_uri = self.cso_repo.canonicalize(topic_uri)
        
        # Determine the graph and logic based on mode
        if mode == "hierarchy":
            graph = self.cso_repo.hierarchy_graph
            # In superTopicOf, u is super of v. (u -> v)
            # Predecessors of v = topics that are broader than v ("up")
            # Successors of v = topics that are narrower than v ("down")
            if direction == "up":
                neighbors_func = graph.predecessors
            elif direction == "down":
                neighbors_func = graph.successors
            else:
                # "both"
                def both_neighbors(n):
                    return set(graph.predecessors(n)).union(set(graph.successors(n)))
                neighbors_func = both_neighbors
        elif mode == "contribution":
            graph = self.cso_repo.contribution_graph
            # Contribution mode considers both directions regardless of "direction" parameter 
            # based on user requirement: "ambil successors DAN predecessors"
            def all_neighbors(n):
                preds = set(graph.predecessors(n)) if n in graph else set()
                succs = set(graph.successors(n)) if n in graph else set()
                return preds.union(succs)
            neighbors_func = all_neighbors
        else:
            raise ValueError(f"Invalid mode: {mode}")

        if canon_uri not in graph:
            # The topic might not have any edges in the specified graph
            # Return just the topic itself
            label = self.cso_repo.get_topic_label(canon_uri)
            return [label] if label else []

        expanded_uris = {canon_uri}
        current_layer = {canon_uri}
        
        for _ in range(hop):
            next_layer = set()
            for node in current_layer:
                if node in graph:
                    neighbors = neighbors_func(node)
                    next_layer.update(neighbors)
            
            # Canonicalize discovered neighbors (they should be, but just to be safe)
            next_layer = {self.cso_repo.canonicalize(n) for n in next_layer}
            expanded_uris.update(next_layer)
            current_layer = next_layer

        # Convert URIs to labels
        labels = []
        for uri in expanded_uris:
            label = self.cso_repo.get_topic_label(uri)
            if label:
                labels.append(label)
                
        logger.debug(f"Topic {canon_uri} expanded to {len(labels)} topics using mode {mode}, hop {hop}.")
        return labels
