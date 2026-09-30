# JTK-ESCO-CSO Mapping Backend

Backend integrasi dan pemetaan kurikulum Jurusan Teknik Komputer (JTK) ke standar kompetensi kerja Eropa (ESCO) serta Computer Science Ontology (CSO) menggunakan SBERT + Cosine Similarity dan Named Entity Recognition (NER).

---

## 🚀 Setup & Instalasi

### 1. Clone & Virtual Environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Konfigurasi Environment

```bash
# Copy template
cp .env.example .env

# Edit .env — isi GROQ_API_KEY dengan API key Anda
```

### 4. Struktur Data

Pastikan file data berada di lokasi berikut:

```
data/raw/esco/
├── occupations_en.csv
├── skills_en.csv
└── occupationSkillRelations_en.csv

data/raw/jtk/
├── PEO_PLO.csv
├── CLO-JTK (Jurusan Teknik Komupter).csv
└── CLO-Campur.csv

data/raw/cso/
└── CSO.br.json
```

### 5. Jalankan Server

```bash
uvicorn app.main:app --reload
```

Server akan berjalan di `http://localhost:8000`. Buka **`http://localhost:8000/docs`** untuk Swagger UI.

---

## 📁 Struktur Folder Project

```
app/
├── main.py              # Entrypoint FastAPI & Lifespan handler
├── config.py            # Settings via pydantic-settings (.env)
├── api/                 # API Route Routers
│   ├── routes_occupation.py   # POST /occupation/match
│   ├── routes_comparison.py   # POST /comparison/esco-skills-vs-plo
│   ├── routes_cso.py          # POST /cso/clo-to-topics
│   └── routes_integration.py  # POST /integration/peo-plo-clo
├── core/                # Fondasi utama
│   ├── embeddings.py    # SBERT singleton, encode(), cosine_sim()
│   └── ner.py           # Skill extraction via Groq LLM / regex fallback
├── services/            # Business Logic Layer
│   ├── occupation_matcher.py  # PEO → ESCO Occupation matching
│   ├── skill_lookup.py        # ESCO Occupation → Skills lookup
│   ├── skill_comparator.py    # ESCO Skills vs PLO JTK comparison
│   ├── clo_mapper.py          # PLO → CLO JTK matching
│   └── cso_topic_matcher.py   # Keyword → CSO Topic matching
├── repositories/        # Data Access Layer
│   ├── esco_repository.py     # Access ESCO CSV
│   ├── jtk_repository.py      # Access JTK Curriculum CSV
│   └── cso_repository.py      # Access CSO Graph Data
├── models/              # Pydantic Schemas
│   ├── schemas.py       # General request & response models
│   └── cso_schemas.py   # CSO module request & response models
└── utils/
    └── text_cleaning.py # Cleaning & normalisasi teks
```

---

## 🌐 API Endpoints Overview

| Tag | Method | Endpoint | Deskripsi |
|---|:---:|---|---|
| **System** | `GET` | `/health` | Health Check status backend |
| **Occupation Matcher** | `POST` | `/occupation/match` | Mencocokkan PEO JTK ke ESCO Occupation |
| **Skill Comparison** | `POST` | `/comparison/esco-skills-vs-plo` | Membandingkan ESCO Skills vs PLO JTK (Covered, Gap, Overskill) |
| **CSO Module** | `POST` | `/cso/clo-to-topics` | Ekstraksi kata kunci CLO (NER) dan pemetaan ke CSO Topics |
| **End-to-End Integration** | `POST` | `/integration/peo-plo-clo` | Integrasi pemetaan menyeluruh PEO $\rightarrow$ ESCO $\rightarrow$ PLO $\rightarrow$ CLO $\rightarrow$ CSO |

---

## 📝 Contoh Request Payload

### 1. End-to-End Integration (`POST /integration/peo-plo-clo`)
```json
{
  "peo_name": "Programmer",
  "threshold": 0
}
```

### 2. CSO Module (`POST /cso/clo-to-topics`)
```json
{
  "clo_id": "CLO-JTK-05"
}
```

### 3. Skill Comparison (`POST /comparison/esco-skills-vs-plo`)
```json
{
  "occupation_uri": "http://data.europa.eu/esco/occupation/f2b15a0e-e65a-438a-affb-29b9d50b77d1",
  "peo_id": "PEO-01",
  "threshold": 0.5
}
```

### 4. Occupation Matcher (`POST /occupation/match`)
```json
{
  "peo_text": "Programmer",
  "top_k": 5
}
```

---

## 🔑 ID Reference

- **PEO**: `PEO-01` (Programmer), `PEO-02` (DBA), `PEO-03` (Software Tester), `PEO-04` (Technical Writer), `PEO-05` (Cloud Engineer), `PEO-06` (Network Engineer)
- **PLO**: `PLO-01-01` s/d `PLO-01-07` (PLO PEO Programmer), dst.
- **CLO**: `CLO-JTK-05` s/d `CLO-JTK-37` (CLO JTK), `CLO-MIX-01` s/d `CLO-MIX-41`

---

## 🧠 Model SBERT
Model embedding yang digunakan: **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`**
- Mendukung Bahasa Indonesia dan Bahasa Inggris
- Ringan (vektor 384 dimensi)
- Di-load sekali saat startup (Singleton pattern)
