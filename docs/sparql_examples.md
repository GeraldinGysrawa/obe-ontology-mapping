# Contoh Query SPARQL untuk JTK Knowledge Graph

Dokumen ini berisi contoh-contoh *query* SPARQL yang dapat dieksekusi terhadap file `data/processed/jtk_ontology.ttl`. Semua *query* menggunakan struktur ontologi yang baru kita bangun (JTK + CSO + ESCO).

---

## 1. Daftar Mata Kuliah

Query ini menampilkan daftar mata kuliah yang diajarkan. Di ontologi JTK, kita belum memasukkan informasi kredit dan semester, jadi kita menampilkan nama mata kuliah.

```sparql
PREFIX jtk: <http://jtk.polban.ac.id/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?course ?courseName
WHERE {
  ?course a jtk:Course ;
          rdfs:label ?courseName .
}
ORDER BY ?courseName
LIMIT 20
```

---

## 2. PLO dan Mata Kuliah Terkait

Karena di ontologi JTK relasi PLO ke Mata Kuliah dihubungkan melalui CLO (`PLO -> supportedByCLO -> CLO -> taughtIn -> Course`), maka *query*-nya menjadi seperti ini:

```sparql
PREFIX jtk: <http://jtk.polban.ac.id/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?ploText ?courseName
WHERE {
  # Ambil PLO
  ?plo a jtk:PLO ;
       jtk:ploText ?ploText ;
       jtk:supportedByCLO ?clo .
       
  # Ambil Course dari CLO tersebut
  ?clo jtk:taughtIn ?course .
  ?course rdfs:label ?courseName .
}
ORDER BY ?ploText
LIMIT 30
```

---

## 3. Course Learning Outcome (CLO)

Query ini menampilkan CLO beserta *label* singkat dan deskripsi utuhnya yang baru saja kita tambahkan.

```sparql
PREFIX jtk: <http://jtk.polban.ac.id/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?clo ?label ?description
WHERE {
  ?clo a jtk:CLO .
  
  OPTIONAL { ?clo rdfs:label ?label }
  OPTIONAL { ?clo jtk:cloText ?description }
}
ORDER BY ?label
LIMIT 20
```

---

## 4. Analisis Gap: CLO dan Standar Industri (ESCO Skills & Occupations)

Ini adalah fitur utama dari perluasan *Knowledge Graph* kita. Query ini menampilkan korelasi antara teks CLO, *ESCO Skill*, dan *ESCO Occupation* (Profesi) berdasarkan *score* kemiripan semantic (SBERT).

```sparql
PREFIX jtk: <http://jtk.polban.ac.id/ontology#>
PREFIX esco_occ: <http://data.europa.eu/esco/occupation/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?cloLabel ?score ?skillLabel ?occupationLabel
WHERE {
  ?clo a jtk:CLO ;
       rdfs:label ?cloLabel ;
       jtk:hasMapping ?mapping .
       
  # Ekstrak data dari Node Mapping perantara
  ?mapping jtk:hasScore ?score ;
           jtk:mappedSkill ?skill .
           
  ?skill rdfs:label ?skillLabel .
  
  # Cari profesi yang membutuhkan skill tersebut
  ?occupation esco_occ:requiresSkill ?skill ;
              rdfs:label ?occupationLabel .
}
ORDER BY DESC(?score)
LIMIT 50
```

---

## 5. Menelusuri Topik Computer Science (CSO) dari Mata Kuliah

Query ini membantu mengecek topik-topik (keywords) *Computer Science* apa saja yang diajarkan pada suatu mata kuliah secara spesifik.

```sparql
PREFIX jtk: <http://jtk.polban.ac.id/ontology#>
PREFIX cso: <https://cso.kmi.open.ac.uk/topics/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?courseName ?topicLabel
WHERE {
  # Filter berdasarkan mata kuliah spesifik (contoh: Rekayasa Perangkat Lunak)
  ?course rdfs:label "Rekayasa Perangkat Lunak"^^<http://www.w3.org/2001/XMLSchema#string> .
  ?clo jtk:taughtIn ?course .
  
  # Telusuri topik CSO yang dipetakan dari CLO tersebut
  ?clo jtk:hasCSOTopic ?topic .
  ?topic rdfs:label ?topicLabel .
}
ORDER BY ?topicLabel
```

---

## 6. End-to-End Traceability (PEO → ESCO Occupation)

Ini adalah *query* paling kuat (Ultimate Query) yang menghubungkan kurikulum paling atas (PEO) hingga ke profesi industri (ESCO Occupation). Berguna untuk menjawab: *"Mata kuliah apa yang diajarkan untuk mencapai profil lulusan tertentu, dan profesi apa yang akan didapatkan?"*

```sparql
PREFIX jtk: <http://jtk.polban.ac.id/ontology#>
PREFIX esco_occ: <http://data.europa.eu/esco/occupation/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?peoLabel ?ploLabel ?cloLabel ?courseName ?score ?occupationLabel
WHERE {
  # 1. Mulai dari PEO
  ?peo a jtk:PEO ;
       rdfs:label ?peoLabel ;
       jtk:hasPLO ?plo .
       
  # 2. Turun ke PLO
  ?plo rdfs:label ?ploLabel ;
       jtk:supportedByCLO ?clo .
       
  # 3. Turun ke CLO dan catat Mata Kuliahnya
  ?clo rdfs:label ?cloLabel ;
       jtk:taughtIn ?course .
  ?course rdfs:label ?courseName .
  
  # 4. Hubungkan CLO dengan Mapping perantara
  ?clo jtk:hasMapping ?mapping .
  ?mapping jtk:hasScore ?score ;
           jtk:mappedSkill ?skill .
           
  # 5. Cari Profesi ESCO yang sesuai
  ?occupation esco_occ:requiresSkill ?skill ;
              rdfs:label ?occupationLabel .
}
ORDER BY ?peoLabel DESC(?score)
LIMIT 100
```

---

## 7. Filter PEO Spesifik (Contoh: "Programmer")

Query ini mirip dengan *End-to-End Traceability*, namun kita menambahkan fungsi `FILTER` agar mesin hanya menampilkan jalur untuk PEO yang mengandung kata kunci tertentu.

```sparql
PREFIX jtk: <http://jtk.polban.ac.id/ontology#>
PREFIX esco_occ: <http://data.europa.eu/esco/occupation/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT ?peoLabel ?ploLabel ?cloLabel ?courseName ?score ?occupationLabel
WHERE {
  # 1. Mulai dari PEO
  ?peo a jtk:PEO ;
       rdfs:label ?peoLabel ;
       jtk:hasPLO ?plo .
       
  # ---> FILTER <---
  # Hanya ambil PEO yang memiliki kata "Programmer" (case-insensitive)
  FILTER(CONTAINS(LCASE(?peoLabel), "programmer"))
       
  # 2. Turun ke PLO
  ?plo rdfs:label ?ploLabel ;
       jtk:supportedByCLO ?clo .
       
  # 3. Turun ke CLO dan catat Mata Kuliahnya
  ?clo rdfs:label ?cloLabel ;
       jtk:taughtIn ?course .
  ?course rdfs:label ?courseName .
  
  # 4. Hubungkan CLO dengan Mapping perantara
  ?clo jtk:hasMapping ?mapping .
  ?mapping jtk:hasScore ?score ;
           jtk:mappedSkill ?skill .
           
  # 5. Cari Profesi ESCO 
  ?occupation esco_occ:requiresSkill ?skill ;
              rdfs:label ?occupationLabel .
}
ORDER BY DESC(?score)
LIMIT 100
```

---

## 8. Ringkasan PEO Programmer → PLO → ESCO Skill & Occupation

Jika Anda hanya ingin melihat **PLO, ESCO Skill, dan ESCO Occupation** khusus untuk profil lulusan **Programmer (Web/ Mobile/ Desktop)** tanpa perlu menampilkan Mata Kuliah dan CLO yang panjang, Anda bisa menggunakan *query* ini. (Kita menggunakan `DISTINCT` agar tabel hasilnya bersih dari baris yang ganda/duplikat).

```sparql
PREFIX jtk: <http://jtk.polban.ac.id/ontology#>
PREFIX esco_occ: <http://data.europa.eu/esco/occupation/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

SELECT DISTINCT ?peoLabel ?ploLabel ?skillLabel ?occupationLabel
WHERE {
  # 1. Mulai dari PEO, ambil spesifik "Programmer (Web/ Mobile/ Desktop)"
  ?peo a jtk:PEO ;
       rdfs:label ?peoLabel ;
       jtk:hasPLO ?plo .
       
  FILTER(CONTAINS(LCASE(?peoLabel), "programmer (web/ mobile/ desktop)"))
       
  # 2. Ambil label PLO
  ?plo rdfs:label ?ploLabel ;
       jtk:supportedByCLO ?clo .
       
  # 3. Telusuri melewati CLO secara tersembunyi (tidak ditampilkan di tabel)
  ?clo jtk:hasMapping ?mapping .
  ?mapping jtk:mappedSkill ?skill .
           
  # 4. Ambil Label dari ESCO Skill dan ESCO Occupation
  ?skill rdfs:label ?skillLabel .
  ?occupation esco_occ:requiresSkill ?skill ;
              rdfs:label ?occupationLabel .
}
ORDER BY ?ploLabel ?occupationLabel
```
