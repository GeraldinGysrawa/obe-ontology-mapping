# JTK-ESCO Mapping Pipeline

Backend pipeline pemetaan kurikulum Jurusan Teknik Komputer (JTK) ke standar kompetensi kerja Eropa (ESCO) menggunakan SBERT + Cosine Similarity.

## Pipeline Overview

```
A (PEO JTK) ──SBERT+Cosine──> C (ESCO Occupation) ──lookup──> D (ESCO Skills)
A ──relasi langsung──> E (PLO JTK)
D <──SBERT+Cosine──> E
D + E ──threshold──> F (Covered) + G (Gap)
E ──3 metode──> K (CLO JTK) ──relasi──> L (Mata Kuliah)
```

## Setup

### 1. Clone & Virtual Environment

```bash
cd jtk-esco-mapping
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

### 4. Siapkan Data

Pastikan file data sudah ada di lokasi yang benar:

```
data/raw/esco/
├── occupations_en.csv
├── skills_en.csv
└── occupationSkillRelations_en.csv

data/raw/jtk/
├── PEO_PLO.csv
├── CLO-JTK (Jurusan Teknik Komupter).csv
└── CLO-Campur.csv
```

### 5. Jalankan Server

```bash
uvicorn app.main:app --reload
```

Server akan berjalan di `http://localhost:8000`. Buka `http://localhost:8000/docs` untuk Swagger UI.

> **Catatan**: Saat pertama kali startup, model SBERT akan didownload (~500MB). Proses ini hanya terjadi sekali.

## Struktur Folder

```
app/
├── main.py              # Entrypoint FastAPI
├── config.py            # Settings via pydantic-settings (.env)
├── api/                 # Endpoint API
│   ├── routes_occupation.py   # POST /occupation/match, GET /occupation/skills
│   ├── routes_comparison.py   # POST /comparison/plo-vs-esco
│   ├── routes_clo.py          # POST /clo/match-from-plo
│   └── routes_experiments.py  # POST /experiments/compare/{plo_id}, /experiments/export
├── core/                # Fondasi yang dipakai semua modul
│   ├── embeddings.py    # SBERT singleton, encode(), cosine_sim(), cache .npy
│   └── ner.py           # NER via Groq LLM + fallback regex
├── services/            # Business logic pipeline utama
│   ├── occupation_matcher.py  # Step A→C
│   ├── skill_lookup.py        # Step C→D
│   ├── skill_comparator.py    # Step D↔E → F/G
│   └── clo_mapper.py          # Step E→K (SBERT default)
├── experiments/         # Modul eksperimen 3 metode
│   ├── schemas.py             # Pydantic model eksperimen
│   ├── method_sbert_only.py   # Metode 1: SBERT saja
│   ├── method_ner_only.py     # Metode 2: NER + fuzzy match
│   ├── method_ner_sbert.py    # Metode 3: NER + SBERT
│   ├── runner.py              # Jalankan 3 metode, log timing
│   └── export_for_review.py   # Export CSV untuk evaluasi manual
├── repositories/        # Data access layer
│   ├── esco_repository.py     # Load & query ESCO CSV
│   └── jtk_repository.py      # Load & query JTK data
├── models/
│   └── schemas.py       # Pydantic request/response models
└── utils/
    └── text_cleaning.py # Normalisasi teks sebelum embedding
```

## API Endpoints

### Pipeline Utama

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/health` | Health check |
| `POST` | `/occupation/match` | Step A→C: Match PEO ke ESCO Occupation |
| `GET` | `/occupation/skills?uri=...` | Step C→D: Ambil ESCO Skills untuk occupation |
| `POST` | `/comparison/plo-vs-esco` | Step D↔E → F/G: Bandingkan PLO vs ESCO Skills |
| `POST` | `/clo/match-from-plo` | Step E→K: Match PLO ke CLO (SBERT default) |

### Eksperimen

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `POST` | `/experiments/compare/{plo_id}` | Jalankan 3 metode untuk satu PLO |
| `POST` | `/experiments/export` | Export hasil ke CSV untuk evaluasi manual |

## Menjalankan Eksperimen

### 1. Cek Cepat via Swagger UI

1. Buka `http://localhost:8000/docs`
2. Cari endpoint `POST /experiments/compare/{plo_id}`
3. Masukkan PLO ID (mis. `PLO-01-01`)
4. Lihat hasil JSON dengan perbandingan 3 metode + timing

### 2. Export CSV untuk Evaluasi Manual

1. Panggil `POST /experiments/export` dengan body:
   ```json
   {"plo_ids": []}
   ```
   (kosong = semua PLO)

2. File CSV akan tersimpan di `data/processed/experiment_review.csv`

3. Buka CSV di Excel, isi kolom:
   - `relevan_manual`: Ya/Tidak — apakah matching relevan menurut evaluator
   - `catatan`: Catatan tambahan

### ID Reference

- **PEO**: `PEO-01` (Programmer), `PEO-02` (DBA), `PEO-03` (Software Tester), `PEO-04` (Technical Writer), `PEO-05` (Cloud Engineer), `PEO-06` (Network Engineer)
- **PLO**: `PLO-01-01` s/d `PLO-01-07` (PLO dari PEO Programmer), dst.
- **CLO**: `CLO-JTK-05` s/d `CLO-JTK-37` (CLO khusus JTK), `CLO-MIX-01` s/d `CLO-MIX-41`

## Tests

```bash
pytest tests/ -v
```

## Model SBERT

Proyek ini menggunakan model **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** secara konsisten di seluruh pipeline. Model ini:
- Multilingual (mendukung Bahasa Indonesia & Inggris)
- Ringan (dimensi 384)
- Cocok untuk semantic similarity task

## Yang Tidak Diimplementasikan (Future Work)

- ❌ Neo4j / GDS / Degree Centrality — fase berikutnya
- ❌ Mapping D→K (ESCO Skills → CLO langsung) — gap granularitas
- ❌ Fusion ESCO+CSO — future work
