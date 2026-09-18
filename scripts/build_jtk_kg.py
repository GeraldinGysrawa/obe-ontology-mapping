import sys
import asyncio
import argparse
from pathlib import Path
import numpy as np

# Tambahkan root direktori proyek ke PYTHONPATH agar bisa import 'app'
sys.path.append(str(Path(__file__).parent.parent))

from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import RDFS, XSD
from urllib.parse import quote

from app.repositories import jtk_repository, esco_repository
from app.repositories.cso_repository import CSORepository
from app.services import clo_mapper
from app.core.embeddings import load_model, encode, load_or_compute_embeddings, cosine_sim
from app.api.routes_cso import _extract_and_match

# Inisiasi CSO repo untuk me-load graf di memory
cso_repo = CSORepository()

async def build_knowledge_graph_async(esco_threshold: float):
    print("Membangun JTK Knowledge Graph...")
    
    print("Loading CSO Repository...")
    cso_repo.load_data()
    
    print("Loading SBERT Model...")
    load_model()
    
    # Pastikan esco dan cso termuat
    esco_data = esco_repository.get_all_skill_texts()
    esco_texts = [f"{label}. {desc}" for _, label, desc in esco_data]
    print("Loading ESCO Embeddings...")
    esco_emb = load_or_compute_embeddings(esco_texts, cache_key="esco_skills")
    
    g = Graph()
    
    JTK = Namespace("http://jtk.polban.ac.id/ontology#")
    ESCO = Namespace("http://data.europa.eu/esco/skill/")
    ESCO_OCC = Namespace("http://data.europa.eu/esco/occupation/")
    CSO = Namespace("https://cso.kmi.open.ac.uk/topics/")
    
    g.bind("jtk", JTK)
    g.bind("esco", ESCO)
    g.bind("esco_occ", ESCO_OCC)
    g.bind("cso", CSO)
    g.bind("rdfs", RDFS)
    
    # ── 1. ESCO Occupations ──────────────────────────────────────────────
    print("Memproses relasi ESCO Occupation -> ESCO Skill...")
    occupations_list = esco_repository.get_all_occupation_labels() # (uri, label)
    
    # Dictionary lookup untuk occupation labels
    occ_label_dict = {uri: label for uri, label in occupations_list}
    
    relations_df = esco_repository.load_relations()
    
    # Masukkan relasi Occupation -> Skill
    for _, row in relations_df.iterrows():
        occ_uri_str = row["occupationUri"]
        skill_uri_str = row["skillUri"]
        
        occ_uri = URIRef(occ_uri_str)
        skill_uri = URIRef(skill_uri_str)
        
        g.add((occ_uri, RDF.type, ESCO_OCC.Occupation))
        if occ_uri_str in occ_label_dict:
            g.add((occ_uri, RDFS.label, Literal(occ_label_dict[occ_uri_str], datatype=XSD.string)))
            
        g.add((occ_uri, ESCO_OCC.requiresSkill, skill_uri))
    
    # ── 2. PEO & PLO ───────────────────────────────────────────────
    peos = jtk_repository.get_peo_list()
    plos = jtk_repository.get_plo_list()
    
    print(f"Memproses {len(peos)} PEO dan {len(plos)} PLO...")
    
    for peo in peos:
        peo_uri = JTK[peo.peo_id.replace("-", "_")]
        g.add((peo_uri, RDF.type, JTK.PEO))
        g.add((peo_uri, RDFS.label, Literal(f"PEO: {peo.peo_text}", datatype=XSD.string)))
        
        for plo in plos:
            if plo.peo_id == peo.peo_id:
                plo_uri = JTK[plo.plo_id.replace("-", "_")]
                g.add((peo_uri, JTK.hasPLO, plo_uri))
                g.add((plo_uri, RDF.type, JTK.PLO))
                g.add((plo_uri, RDFS.label, Literal(f"PLO: {plo.plo_text[:30]}...", datatype=XSD.string)))
                g.add((plo_uri, JTK.ploText, Literal(plo.plo_text, datatype=XSD.string)))
                
                # SBERT PLO -> CLO (Threshold > 0.5)
                matches = clo_mapper.match_plo_to_clo(plo.plo_id, top_k=3, clo_source="jtk")
                for match in matches.matched_clo:
                    if match.score > 0.5:
                        clo_uri = JTK[match.clo_id.replace("-", "_")]
                        g.add((plo_uri, JTK.supportedByCLO, clo_uri))
    
    # ── 3. CLO & Mapping ke CSO / ESCO ───────────────────────────────────────
    clos = jtk_repository.get_all_clo(source="jtk")
    print(f"Memproses {len(clos)} CLO dan mapping ke CSO & ESCO...")
    
    for i, clo in enumerate(clos):
        print(f"[{i+1}/{len(clos)}] Memproses CLO: {clo.clo_id}")
        clo_uri = JTK[clo.clo_id.replace("-", "_")]
        
        g.add((clo_uri, RDF.type, JTK.CLO))
        g.add((clo_uri, RDFS.label, Literal(f"CLO: {clo.clo_id}", datatype=XSD.string)))
        g.add((clo_uri, JTK.cloText, Literal(clo.clo_text, datatype=XSD.string)))
        
        if clo.mata_kuliah:
            safe_mk_name = quote(clo.mata_kuliah.replace(" ", "_"))
            course_uri = JTK[f"Course_{safe_mk_name}"]
            g.add((clo_uri, JTK.taughtIn, course_uri))
            g.add((course_uri, RDF.type, JTK.Course))
            g.add((course_uri, RDFS.label, Literal(clo.mata_kuliah, datatype=XSD.string)))
            
        # a. NER + CSO Mapping
        keywords, matched_topics = await _extract_and_match(clo.clo_text, top_k_per_keyword=1)
        topic_labels = []
        for m in matched_topics:
            topic_uri = URIRef(m.topic_uri)
            g.add((clo_uri, JTK.hasCSOTopic, topic_uri))
            g.add((topic_uri, RDF.type, CSO.Topic))
            g.add((topic_uri, RDFS.label, Literal(m.topic_name, datatype=XSD.string)))
            topic_labels.append(m.topic_name)
            
        # b. ESCO Mapping (dari CSO Topics yang diekstrak, cari ESCO Skills)
        if topic_labels:
            cso_emb = encode(topic_labels, clean=False)
            sim_matrix = cosine_sim(cso_emb, esco_emb)
            
            for t_idx, topic_label in enumerate(topic_labels):
                # Ambil skill ESCO yang paling cocok
                best_idx = int(np.argmax(sim_matrix[t_idx]))
                best_score = float(sim_matrix[t_idx][best_idx])
                
                if best_score >= esco_threshold:
                    best_uri, best_label, _ = esco_data[best_idx]
                    skill_uri = URIRef(best_uri)
                    
                    # Create Intermediate Mapping Node
                    map_id = f"Mapping_{clo.clo_id.replace('-', '_')}_ESCO_{t_idx}"
                    map_uri = JTK[map_id]
                    
                    g.add((clo_uri, JTK.hasMapping, map_uri))
                    g.add((map_uri, RDF.type, JTK.ESCOMapping))
                    g.add((map_uri, JTK.mappedSkill, skill_uri))
                    g.add((map_uri, JTK.hasScore, Literal(best_score, datatype=XSD.float)))
                    
                    g.add((skill_uri, RDF.type, ESCO.Skill))
                    g.add((skill_uri, RDFS.label, Literal(best_label, datatype=XSD.string)))

    
    # ── 4. Export Graph ke RDF/Turtle ────────────────────────────────────────
    output_path = Path("data/processed/jtk_ontology.ttl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Menyimpan graph ke file...")
    g.serialize(destination=str(output_path), format="turtle")
    
    print("="*60)
    print(f"Knowledge Graph berhasil dibuat! Tersimpan di: {output_path}")
    print(f"Total Triples (Axioms): {len(g)}")
    print("="*60)


def build_knowledge_graph(threshold: float):
    asyncio.run(build_knowledge_graph_async(threshold))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build JTK Knowledge Graph")
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.65, 
        help="SBERT Cosine Similarity threshold for ESCO Mapping (default: 0.65)"
    )
    args = parser.parse_args()
    
    build_knowledge_graph(args.threshold)
