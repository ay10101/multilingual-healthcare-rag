# 🩺 Multilingual Healthcare Misinformation RAG

A clinical misinformation fact-checking system leveraging **Retrieval-Augmented Generation (RAG)** across **English**, **Hindi (हिंदी)**, and **Hinglish**.

This project provides both:
1. **`Multilingual_Healthcare_Misinformation_RAG.ipynb`**: An interactive Google Colab notebook covering Steps 1 through 10.
2. **Modular Python Workspace**: Clean, production-ready source code (`src/`), curated evidence dataset (`data/`), and test suite (`tests/`).

---

## 🌟 Project Highlights

- **Multilingual Semantic Embeddings**: Uses `paraphrase-multilingual-MiniLM-L12-v2` to map claims in Hindi or Hinglish (e.g., *"Haldi doodh cancer ko theek karta hai"*) to English clinical evidence (e.g., Cancer Research UK findings on turmeric and curcumin).
- **Fast Dense Vector Search**: Inner-product FAISS index (`IndexFlatIP`) for exact cosine similarity matching.
- **Domain Guardrail**: Early rule-based classification that filters out non-health claims (e.g., *"India won the cricket match"*) before vector retrieval.
- **Explainable Verdict & Risk Assessment**: Determines whether claims are `Supported`, `Refuted`, `Uncertain`, or have `No relevant evidence`, mapping them to risk tiers (`Low`, `Medium`, `High`, `Very High`).
- **Educational Templates & Compliance**: Includes disclaimers and clickable citations from reputable sources (**WHO, CDC, NHS, Cancer Research UK, PubMed**).
- **Interactive UI**: Gradio web interface with one-click presentation demo test claims.

---

## 📁 Repository Structure

```
multilingual-healthcare-rag/
├── Multilingual_Healthcare_Misinformation_RAG.ipynb  # Primary Google Colab Notebook (Steps 1 to 10)
├── README.md                                          # Documentation & Usage Guide
├── requirements.txt                                  # Dependency specifications
├── data/
│   ├── Datensatz_raw.csv                              # Original Datensatz source dataset (750 records)
│   └── evidence_dataset.csv                           # 750 standardized clinical evidence records with URLs
├── src/
│   ├── __init__.py
│   ├── preprocessing.py                              # Text cleaning & Hinglish normalization
│   ├── language_detector.py                          # Hindi (Devanagari) / Hinglish / English detector
│   ├── health_classifier.py                          # Rule-based health claim filter
│   ├── rag_engine.py                                 # SentenceTransformer & FAISS vector index
│   ├── verdict_engine.py                             # Verdict evaluation, risk rating, & explanation
│   └── app.py                                        # Gradio web application
├── scripts/
│   └── generate_notebook.py                          # Notebook generator script
└── tests/
    └── test_pipeline.py                              # Automated unit and integration test suite
```

---

## 🚀 Running on Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File** ➔ **Upload notebook** and select `Multilingual_Healthcare_Misinformation_RAG.ipynb`.
3. *(Optional)* Turn on GPU acceleration: **Runtime** ➔ **Change runtime type** ➔ **T4 GPU**. (Runs on CPU if GPU is unavailable).
4. Run cells sequentially or click **Runtime** ➔ **Run all**.
5. The final cell will launch the interactive Gradio web interface with a shareable public URL.

---

## 💻 Running Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Gradio App
```bash
python3 src/app.py
```
Open your browser at `http://127.0.0.1:7860`.

### 3. Run the Test Suite
```bash
python3 tests/test_pipeline.py
```

---

## 📊 Dataset Schema (`data/evidence_dataset.csv`)

To add or update records, provide a CSV with the following columns:

| Column | Type | Description | Example |
| :--- | :--- | :--- | :--- |
| `id` | integer | Unique identifier | `1` |
| `claim_topic` | string | Clinical topic or myth category | `cancer cure` |
| `evidence_text` | string | Verified medical consensus snippet | `There is no reliable scientific evidence that turmeric alone cures cancer.` |
| `stance` | string | Stance towards the misinformation (`refutes` / `supports`) | `refutes` |
| `source` | string | Trusted publisher | `Cancer Research UK` |
| `url` | string | Direct link to official evidence page | `https://www.cancerresearchuk.org/...` |

---

## 🧪 Evaluation Test Claims

| # | Demo Claim | Input Language | Health? | Verdict | Risk |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | `Haldi doodh cancer ko theek karta hai` | Hinglish | Yes | **Refuted** | High |
| 2 | `Vaccines autism cause karte hain` | Hinglish | Yes | **Refuted** | High |
| 3 | `Smoking causes lung cancer` | English | Yes | **Supported** | Low |
| 4 | `Antibiotics cold ko cure karte hain` | Hinglish | Yes | **Refuted** | High |
| 5 | `India won the cricket match` | English | **No** | **Non-Health Claim** | None |
