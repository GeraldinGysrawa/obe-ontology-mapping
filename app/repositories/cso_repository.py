import pandas as pd
import networkx as nx
import logging
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)

class CSORepository:
    _instance = None
    _is_loaded = False

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CSORepository, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self, cso_path: str = "data/raw/cso/CSO.3.5.csv"):
        if not self._is_loaded:
            self.cso_path = cso_path
            
            # Data structures
            self.label_lookup: Dict[str, str] = {}
            self.equivalence_lookup: Dict[str, str] = {}
            self.hierarchy_graph = nx.DiGraph()
            self.contribution_graph = nx.DiGraph()
            
            self._is_loaded = True

    def load_data(self):
        if self.label_lookup:
            return # Already loaded
            
        logger.info(f"Loading CSO data from {self.cso_path}...")
        
        if not os.path.exists(self.cso_path):
            logger.error(f"CSO file not found at {self.cso_path}")
            return
            
        # Read CSV without header
        df = pd.read_csv(self.cso_path, header=None, names=['subject', 'predicate', 'object'])
        
        # Clean the < and > around URIs
        for col in ['subject', 'predicate', 'object']:
            df[col] = df[col].astype(str).str.strip('<>')
            
        # Separate data by predicates
        df_label = df[df['predicate'].str.contains('rdf-schema#label')]
        df_eq = df[df['predicate'].str.contains('cso#preferentialEquivalent')]
        df_hierarchy = df[df['predicate'].str.contains('cso#superTopicOf')]
        df_contrib = df[df['predicate'].str.contains('cso#contributesTo')]
        
        # 1. Equivalence lookup
        self.equivalence_lookup = dict(zip(df_eq['subject'], df_eq['object']))
        logger.info(f"Loaded {len(self.equivalence_lookup)} preferential equivalence links.")
        
        # 2. Label lookup (apply canonicalization to subjects)
        raw_labels = dict(zip(df_label['subject'], df_label['object']))
        for subj, label in raw_labels.items():
            canon_subj = self.canonicalize(subj)
            # if multiple map to same canonical, just use the first/any
            if canon_subj not in self.label_lookup:
                # Clean quotes and RDF language tag (@en .)
                clean_label = label.strip('"').strip()
                # Remove trailing "@en ." or "@en"
                if clean_label.endswith('@en .'):
                    clean_label = clean_label[:-5].strip()
                elif clean_label.endswith('@en'):
                    clean_label = clean_label[:-3].strip()
                self.label_lookup[canon_subj] = clean_label
                
        logger.info(f"Loaded {len(self.label_lookup)} canonical labels.")
        
        # 3. Hierarchy Graph (superTopicOf)
        for _, row in df_hierarchy.iterrows():
            canon_subj = self.canonicalize(row['subject'])
            canon_obj = self.canonicalize(row['object'])
            self.hierarchy_graph.add_edge(canon_subj, canon_obj, predicate="superTopicOf")
            
        logger.info(f"Loaded Hierarchy Graph: {self.hierarchy_graph.number_of_nodes()} nodes, {self.hierarchy_graph.number_of_edges()} edges.")
        
        # 4. Contribution Graph (contributesTo)
        for _, row in df_contrib.iterrows():
            canon_subj = self.canonicalize(row['subject'])
            canon_obj = self.canonicalize(row['object'])
            self.contribution_graph.add_edge(canon_subj, canon_obj, predicate="contributesTo")
            
        logger.info(f"Loaded Contribution Graph: {self.contribution_graph.number_of_nodes()} nodes, {self.contribution_graph.number_of_edges()} edges.")

    def canonicalize(self, uri: str) -> str:
        """Return the preferential equivalent URI if exists, else the URI itself."""
        return self.equivalence_lookup.get(uri, uri)

    def get_topic_label(self, uri: str) -> Optional[str]:
        """Return the clean label for a given URI."""
        canon_uri = self.canonicalize(uri)
        return self.label_lookup.get(canon_uri)
        
    def get_all_topic_names(self) -> List[str]:
        """Return all unique, clean topic labels."""
        return list(self.label_lookup.values())
