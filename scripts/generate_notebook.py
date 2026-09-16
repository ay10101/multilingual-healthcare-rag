"""
Script to generate the complete Google Colab Notebook:
Multilingual_Healthcare_Misinformation_RAG.ipynb
"""

import json
import os

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {
            "name": "Multilingual_Healthcare_Misinformation_RAG.ipynb",
            "provenance": []
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": []
}

def add_md(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    })

def add_code(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source if isinstance(source, list) else [line + "\n" for line in source.split("\n")]
    })

# Header
add_md("""# 🩺 Multilingual Healthcare Misinformation Detection & Verification (RAG)
### Cross-Lingual Clinical Fact-Checking in English, Hindi, and Hinglish using Dense Retrieval

This Google Colab notebook implements an end-to-end Retrieval-Augmented Generation (RAG) system designed to detect and refute healthcare misinformation in multilingual environments (English, Hindi, and Hinglish).

---

### Key Workflow (10 Steps)
1. **Environment Setup & Hardware Detection**: Auto-detects GPU (CUDA) or CPU.
2. **Library Installation**: `sentence-transformers`, `faiss-cpu`, `langdetect`, `pandas`, `scikit-learn`, `gradio`.
3. **Trusted Evidence Dataset**: 25+ curated records from WHO, CDC, NHS, Cancer.org, and PubMed.
4. **Text Preprocessing**: Handles Devanagari Hindi Unicode & Hinglish colloquialisms.
5. **Language Identification**: Tri-lingual detection (Hindi, Hinglish, English).
6. **Health Domain Classification**: Filters non-health statements before retrieval.
7. **Cross-Lingual Embeddings & FAISS Vector Index**: Using `paraphrase-multilingual-MiniLM-L12-v2`.
8. **Verdict & Risk Assessment**: Multi-tier stance and health hazard evaluation.
9. **Explanation & Structured Output**: Standardized educational templates and disclaimers.
10. **Interactive Gradio Interface**: Live web demo with presentation test cases.""")

# Step 1
add_md("""## Step 1 — Hardware Configuration & Colab Setup
Check available hardware acceleration. The multilingual embedding model runs efficiently on CPU, but automatically leverages GPU (CUDA) if enabled in Colab (*Runtime -> Change runtime type -> T4 GPU*).""")

add_code("""import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✅ Hardware Acceleration Device: {device.upper()}")
if device == "cuda":
    print(f"   GPU Model: {torch.cuda.get_device_name(0)}")
else:
    print("   Running on CPU (Fully supported for this model and dataset size)")""")

# Step 2
add_md("""## Step 2 — Install Required Libraries
Install the core NLP and RAG libraries:
- `sentence-transformers`: Multilingual dense text embeddings
- `faiss-cpu`: Fast vector similarity search
- `langdetect`: English/Hindi language detection
- `pandas` & `scikit-learn`: Data handling and similarity utilities
- `gradio`: Web interface for live evaluation""")

add_code("""!pip -q install sentence-transformers faiss-cpu langdetect pandas scikit-learn gradio

import re
import unicodedata
import numpy as np
import pandas as pd
import faiss
import gradio as gr
from sentence_transformers import SentenceTransformer

print("✅ All core libraries successfully imported!")""")

# Step 3
add_md("""## Step 3 — Prepare Trusted Evidence Dataset
We create a curated clinical repository containing verified evidence records from reputable health organizations (**WHO, CDC, NHS, Memorial Sloan Kettering, Cancer Research UK, PubMed**).

Required Schema:
- `id`: Unique record identifier
- `claim_topic`: The clinical subject / myth category
- `evidence_text`: Verified medical consensus statement
- `stance`: `supports` or `refutes`
- `source`: Reputable institutional publisher
- `url`: Direct source link for transparency""")

add_code("""evidence_data = [
    {
        "id": 1,
        "claim_topic": "cancer cure",
        "evidence_text": "There is no reliable scientific evidence that turmeric (haldi), turmeric milk (haldi doodh), or curcumin alone can cure or treat cancer in humans.",
        "stance": "refutes",
        "source": "Cancer Research UK",
        "url": "https://www.cancerresearchuk.org/about-cancer/treatment/complementary-alternative-therapies/individual-therapies/turmeric"
    },
    {
        "id": 2,
        "claim_topic": "cancer cure",
        "evidence_text": "While curcumin has shown antioxidant properties in laboratory studies, clinical trials have not demonstrated that turmeric cures malignant tumors.",
        "stance": "refutes",
        "source": "Memorial Sloan Kettering",
        "url": "https://www.mskcc.org/cancer-care/integrative-medicine/herbs/turmeric"
    },
    {
        "id": 3,
        "claim_topic": "smoking and cancer",
        "evidence_text": "Cigarette smoking is the leading cause of lung cancer, linked to approximately 80% to 90% of lung cancer deaths.",
        "stance": "supports",
        "source": "CDC",
        "url": "https://www.cdc.gov/tobacco/basic_information/health_effects/cancer/index.htm"
    },
    {
        "id": 4,
        "claim_topic": "smoking and cancer",
        "evidence_text": "Tobacco smoke contains more than 7,000 chemicals, at least 69 of which are known to cause cancer in humans and animals.",
        "stance": "supports",
        "source": "WHO",
        "url": "https://www.who.int/news-room/fact-sheets/detail/tobacco"
    },
    {
        "id": 5,
        "claim_topic": "vaccines",
        "evidence_text": "Multiple rigorous scientific studies involving millions of children over decades have found no causal link between vaccines and autism.",
        "stance": "refutes",
        "source": "WHO",
        "url": "https://www.who.int/news-room/questions-and-answers/item/vaccines-and-immunization-myths-and-facts"
    },
    {
        "id": 6,
        "claim_topic": "vaccines",
        "evidence_text": "Extensive scientific research by the Institute of Medicine has demonstrated that vaccines and vaccine ingredients do not cause autism.",
        "stance": "refutes",
        "source": "CDC",
        "url": "https://www.cdc.gov/vaccinesafety/concerns/autism.html"
    },
    {
        "id": 7,
        "claim_topic": "vaccines",
        "evidence_text": "Vaccines are rigorously evaluated for safety and provide high efficacy in preventing life-threatening infectious diseases.",
        "stance": "supports",
        "source": "WHO",
        "url": "https://www.who.int/health-topics/vaccines-and-immunization"
    },
    {
        "id": 8,
        "claim_topic": "antibiotics",
        "evidence_text": "Antibiotics treat bacterial infections only; they do not kill viruses or cure viral infections such as the common cold, flu, or viral bronchitis.",
        "stance": "refutes",
        "source": "CDC",
        "url": "https://www.cdc.gov/antibiotic-use/common-illnesses.html"
    },
    {
        "id": 9,
        "claim_topic": "antibiotics",
        "evidence_text": "Taking antibiotics for viral illnesses will not cure the infection, will not prevent others from getting sick, and can cause unnecessary side effects.",
        "stance": "refutes",
        "source": "NHS",
        "url": "https://www.nhs.uk/conditions/antibiotics/"
    },
    {
        "id": 10,
        "claim_topic": "antibiotics",
        "evidence_text": "Antibiotic overuse for viral infections accelerates antibiotic resistance, rendering crucial medicines ineffective against serious bacterial threats.",
        "stance": "refutes",
        "source": "WHO",
        "url": "https://www.who.int/news-room/fact-sheets/detail/antibiotic-resistance"
    },
    {
        "id": 11,
        "claim_topic": "diabetes",
        "evidence_text": "There is currently no permanent herbal cure for diabetes; diabetes is a chronic condition managed through lifestyle, insulin, and prescription medication.",
        "stance": "refutes",
        "source": "American Diabetes Association",
        "url": "https://diabetes.org/about-diabetes"
    },
    {
        "id": 12,
        "claim_topic": "diabetes",
        "evidence_text": "While some herbs like bitter gourd or fenugreek may modestly assist glucose metabolism, no herbal medicine reverses or permanently cures diabetes.",
        "stance": "refutes",
        "source": "Mayo Clinic",
        "url": "https://www.mayoclinic.org/diseases-conditions/diabetes/in-depth/diabetes-management/art-20047945"
    },
    {
        "id": 13,
        "claim_topic": "diabetes",
        "evidence_text": "Discontinuing medical therapy for diabetes in favor of unproven alternative or herbal remedies can lead to diabetic ketoacidosis and organ failure.",
        "stance": "refutes",
        "source": "NHS",
        "url": "https://www.nhs.uk/conditions/type-2-diabetes/"
    },
    {
        "id": 14,
        "claim_topic": "hygiene",
        "evidence_text": "Handwashing with soap and clean water removes pathogens from hands and reduces the transmission of respiratory and diarrheal infections.",
        "stance": "supports",
        "source": "CDC",
        "url": "https://www.cdc.gov/clean-hands/about/index.html"
    },
    {
        "id": 15,
        "claim_topic": "hygiene",
        "evidence_text": "Effective hand hygiene practices in healthcare and community environments can reduce the spread of infectious diseases by up to 50%.",
        "stance": "supports",
        "source": "WHO",
        "url": "https://www.who.int/campaigns/world-hand-hygiene-day"
    },
    {
        "id": 16,
        "claim_topic": "covid-19",
        "evidence_text": "Drinking, gargling, or rinsing with salt water or vinegar does not kill the coronavirus and does not prevent COVID-19 infection.",
        "stance": "refutes",
        "source": "WHO",
        "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters"
    },
    {
        "id": 17,
        "claim_topic": "covid-19",
        "evidence_text": "Inhaling steam or taking extremely hot baths does not prevent or cure coronavirus infection and can cause severe thermal burns to airways.",
        "stance": "refutes",
        "source": "WHO",
        "url": "https://www.who.int/emergencies/diseases/novel-coronavirus-2019/advice-for-public/myth-busters"
    },
    {
        "id": 18,
        "claim_topic": "heart disease",
        "evidence_text": "Regular physical exercise, maintaining a healthy weight, and eating a balanced diet significantly reduce cardiovascular disease risk.",
        "stance": "supports",
        "source": "American Heart Association",
        "url": "https://www.heart.org/en/healthy-living/fitness"
    },
    {
        "id": 19,
        "claim_topic": "hypertension",
        "evidence_text": "Garlic supplements and herbal teas cannot substitute for prescribed antihypertensive medication for controlling severe hypertension.",
        "stance": "refutes",
        "source": "PubMed",
        "url": "https://pubmed.ncbi.nlm.nih.gov/24059423/"
    },
    {
        "id": 20,
        "claim_topic": "fever",
        "evidence_text": "Fever is an immune response to fight infection; mild viral fevers resolve with hydration and rest, and antibiotics provide no benefit for viral fevers.",
        "stance": "refutes",
        "source": "NHS",
        "url": "https://www.nhs.uk/conditions/fever-in-adults/"
    },
    {
        "id": 21,
        "claim_topic": "cancer cure",
        "evidence_text": "No alternative therapies, herbal concoctions, or alkaline diets have been proven to cure or eradicate malignant cancer tumors.",
        "stance": "refutes",
        "source": "Tata Memorial Centre",
        "url": "https://tmc.gov.in/"
    },
    {
        "id": 22,
        "claim_topic": "pregnancy",
        "evidence_text": "Pregnant women should always consult a licensed doctor before taking any herbal remedies, as many can induce premature contractions or toxicity.",
        "stance": "supports",
        "source": "CDC",
        "url": "https://www.cdc.gov/pregnancy/meds/treatingfortwo/index.html"
    },
    {
        "id": 23,
        "claim_topic": "covid-19",
        "evidence_text": "Authorized COVID-19 vaccines have been shown to be safe and highly effective in preventing hospitalization and severe disease outcomes.",
        "stance": "supports",
        "source": "CDC",
        "url": "https://www.cdc.gov/coronavirus/2019-ncov/vaccines/effectiveness.html"
    },
    {
        "id": 24,
        "claim_topic": "mental health",
        "evidence_text": "Depression is a legitimate medical disorder treated effectively with psychotherapy and prescription medication, not a lack of mental resolve.",
        "stance": "supports",
        "source": "NIMH",
        "url": "https://www.nimh.nih.gov/health/topics/depression"
    },
    {
        "id": 25,
        "claim_topic": "immune boost",
        "evidence_text": "No single food, vitamin megadose, or herbal supplement can instantly 'boost' the immune system to make a person immune to viral infections.",
        "stance": "refutes",
        "source": "Harvard Health",
        "url": "https://www.health.harvard.edu/staying-healthy/how-to-boost-your-immune-system"
    }
]

evidence_df = pd.DataFrame(evidence_data)
print(f"✅ Loaded {len(evidence_df)} trusted evidence records across {evidence_df['claim_topic'].nunique()} health topics.")
evidence_df.head(5)""")

# Step 4
add_md("""## Step 4 — Text Preprocessing
The `preprocess_text()` function:
1. Converts text to lowercase.
2. Strips URLs, mentions, hashtags, and excess whitespaces.
3. Preserves Hindi Devanagari script (`\\u0900-\\u097F`).
4. Normalizes common Hinglish spelling variants (e.g. *thik* -> *theek*, *ilaaj* -> *ilaj*, *dawa* -> *dawai*, *karta h* -> *karta hai*).""")

add_code("""HINGLISH_NORMALIZATION_MAP = {
    r"\\bthik\\b": "theek",
    r"\\bteek\\b": "theek",
    r"\\bilaaj\\b": "ilaj",
    r"\\bdawa\\b": "dawai",
    r"\\bdawaai\\b": "dawai",
    r"\\bdawayi\\b": "dawai",
    r"\\bh\\b": "hai",
    r"\\bkartey\\b": "karte",
    r"\\bkarti\\b": "karta",
    r"\\bpeeney\\b": "peena",
    r"\\bpeene\\b": "peena",
    r"\\bkhatm\\b": "khatam",
    r"\\bmadhumeh\\b": "diabetes",
    r"\\bbimaari\\b": "bimari"
}

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"https?://\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"[@#]\\S+", " ", text)
    text = re.sub(r"[^\\w\\s\\u0900-\\u097F\\.,?!-]", " ", text)
    for pattern, replacement in HINGLISH_NORMALIZATION_MAP.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\\s+", " ", text).strip()
    return text

# Test cases from requirements
test_claims = [
    "Haldi doodh cancer ko theek karta hai",
    "Vaccines autism cause karte hain",
    "Antibiotics viral fever ko cure kar dete hain",
    "Smoking causes lung cancer"
]

print("--- Preprocessing Verification ---")
for tc in test_claims:
    print(f"Original: {tc} -> Cleaned: {preprocess_text(tc)}")""")

# Step 5
add_md("""## Step 5 — Language Detection
We implement a hybrid language detector:
- **Hindi**: Identified if Devanagari script characters (`\\u0900-\\u097F`) are present.
- **Hinglish**: Identified if Latin script contains common Romanized Hindi markers (`hai`, `kya`, `karta`, `ke`, `mein`, `dawai`, `ko`, `se`, `nahi`, etc.).
- **English**: Fallback via `langdetect` or default.""")

add_code("""HINGLISH_KEYWORDS = {
    "hai", "hain", "kya", "karta", "karti", "karte", "ke", "ki", "ka", "ko",
    "mein", "me", "se", "par", "bhi", "nahi", "nahin", "dawai", "dawa",
    "theek", "thik", "bimari", "ilaj", "kare", "karo", "peena", "peene",
    "khatam", "hota", "hoti", "hote", "doodh", "haldi", "ghee", "paani"
}

def detect_language(text: str) -> str:
    if not text or not isinstance(text, str):
        return "English"
    if re.search(r"[\\u0900-\\u097F]", text):
        return "Hindi"
    clean_words = set(re.findall(r"\\b[a-z]+\\b", text.lower()))
    if clean_words.intersection(HINGLISH_KEYWORDS):
        return "Hinglish"
    try:
        from langdetect import detect
        detected = detect(text)
        return "Hindi" if detected == "hi" else "English"
    except Exception:
        return "English"

print("--- Language Detection Test ---")
print("1. 'हल्दी दूध कैंसर ठीक करता है' ->", detect_language("हल्दी दूध कैंसर ठीक करता है"))
print("2. 'Haldi doodh cancer ko theek karta hai' ->", detect_language("Haldi doodh cancer ko theek karta hai"))
print("3. 'Smoking causes lung cancer' ->", detect_language("Smoking causes lung cancer"))""")

# Step 6
add_md("""## Step 6 — Health-Related Claim Classification
Before passing queries to vector search, we filter out non-health claims.
- **Health keywords**: `cancer`, `vaccine`, `medicine`, `doctor`, `hospital`, `disease`, `diabetes`, `fever`, `antibiotic`, `covid`, `health`, `treatment`, `cure`, `virus`, `cough`, `infection`, `pregnancy`, `blood`, `heart`, `smoking`, plus Hindi/Hinglish clinical terms (`bimari`, `ilaj`, `dawai`, `haldi`, `dard`, `sehat`).
- *Note*: As planned future work, this rule-based classifier can be upgraded to a fine-tuned Clinical BioBERT / MuRIL model.""")

add_code("""HEALTH_KEYWORDS = {
    "cancer", "vaccine", "vaccines", "vaccination", "medicine", "medicines",
    "doctor", "doctors", "hospital", "hospitals", "disease", "diseases",
    "diabetes", "fever", "antibiotic", "antibiotics", "covid", "covid-19",
    "health", "treatment", "cure", "cures", "curing", "virus", "viral",
    "cough", "infection", "infections", "pregnancy", "pregnant", "blood", "heart",
    "smoking", "tobacco", "cigarette", "lungs", "lung", "hypertension",
    "stroke", "tumor", "chemo", "chemotherapy", "dosage", "drug", "drugs",
    "pain", "autism", "immune", "immunity", "handwash", "handwashing", "soap",
    "hygiene", "sanitize", "sanitizer", "bimari", "bimaari", "ilaj", "ilaaj",
    "dawa", "dawai", "haldi", "doodh", "dard", "sehat", "aspatal", "khansi",
    "कैंसर", "वैक्सीन", "टीका", "दवा", "इलाज", "बीमारी", "मधुमेह", "बुखार"
}

def is_health_related(text: str) -> bool:
    if not text or not isinstance(text, str):
        return False
    tokens = set(re.findall(r"[\\w\\u0900-\\u097F]+", text.lower()))
    if tokens.intersection(HEALTH_KEYWORDS):
        return True
    text_lower = text.lower()
    return any(kw in text_lower for kw in HEALTH_KEYWORDS)

print("--- Health Classification Tests ---")
print("1. 'Turmeric cures cancer' ->", is_health_related("Turmeric cures cancer"), "(Expected: True)")
print("2. 'India won the cricket match' ->", is_health_related("India won the cricket match"), "(Expected: False)")""")

# Step 7
add_md("""## Step 7 — Multilingual Embeddings & FAISS Vector Index
We employ `SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")`:
- Supports 50+ languages and maps concepts across English, Hindi, and Hinglish into a shared embedding space.
- Enables queries like `“Haldi doodh cancer ko cure karta hai”` to retrieve English clinical evidence regarding turmeric and cancer.
- Embeddings are unit-normalized and stored in a FAISS `IndexFlatIP` (Inner Product = Cosine Similarity).""")

add_code("""print("Loading SentenceTransformer model (paraphrase-multilingual-MiniLM-L12-v2)...")
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device=device)

# Encode evidence text with normalization
evidence_texts = evidence_df["evidence_text"].tolist()
evidence_embeddings = embed_model.encode(
    evidence_texts,
    normalize_embeddings=True,
    show_progress_bar=True,
    device=device
)
evidence_embeddings = np.array(evidence_embeddings, dtype=np.float32)

# Build FAISS Index (IndexFlatIP for exact cosine similarity)
dim = evidence_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dim)
faiss_index.add(evidence_embeddings)
print(f"✅ FAISS index created with {faiss_index.ntotal} vectors of dimension {dim}.")

def retrieve_evidence(claim: str, top_k: int = 3):
    cleaned = preprocess_text(claim)
    q_vec = embed_model.encode([cleaned], normalize_embeddings=True, device=device)
    q_vec = np.array(q_vec, dtype=np.float32)
    scores, indices = faiss_index.search(q_vec, min(top_k, len(evidence_df)))
    
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        row = evidence_df.iloc[idx].to_dict()
        row["similarity_score"] = float(round(float(score), 4))
        row["rank"] = rank + 1
        results.append(row)
    return results

# Test retrieval with Hinglish claim
sample_retrieval = retrieve_evidence("Haldi doodh cancer ko theek karta hai", top_k=2)
for r in sample_retrieval:
    print(f"Rank {r['rank']} [Sim: {r['similarity_score']}] ({r['source']}): {r['evidence_text']}")""")

# Step 8
add_md("""## Step 8 — Verdict and Risk Assessment Logic
Transparent verification rules:
- **Refuted**: Closest evidence stance is `refutes` and similarity is high.
- **Supported**: Closest evidence stance is `supports` and similarity is high.
- **Uncertain / Insufficient evidence**: Moderate similarity or conflicting stances among retrieved sources.
- **No relevant evidence**: Low similarity score (< 0.35).

Risk Mapping:
- `Supported` -> **Low Risk**
- `Uncertain` -> **Medium Risk**
- `Refuted` -> **High Risk**
- `Refuted + dangerous medical advice` (e.g. stopping chemotherapy, skipping vaccines, abandoning insulin) -> **Very High Risk**""")

add_code("""DANGEROUS_PATTERNS = [
    "stop chemo", "skip chemo", "stop insulin", "avoid vaccine", "dawai band",
    "chemo mat lo", "insulin mat lo", "teeka mat lagao", "toxic", "bleach"
]

HIGH_SIMILARITY_THRESHOLD = 0.35
MODERATE_SIMILARITY_THRESHOLD = 0.22

def assess_verdict_and_risk(claim: str, retrieved_evidence):
    if not retrieved_evidence:
        return {
            "verdict": "No relevant evidence",
            "evidence_status": "No matching clinical documents found",
            "risk": "Medium",
            "confidence": 0.0,
            "stance": "none"
        }

    top = retrieved_evidence[0]
    top_score = top.get("similarity_score", 0.0)
    top_stance = top.get("stance", "").strip().lower()
    top_topic = top.get("claim_topic", "").strip().lower()

    # Detect conflict among competitively close matches on the same topic
    has_conflict = any(
        item.get("claim_topic", "").strip().lower() == top_topic
        and item.get("stance", "").strip().lower() != top_stance
        and (top_score - item.get("similarity_score", 0.0)) <= 0.12
        and item.get("similarity_score", 0.0) >= MODERATE_SIMILARITY_THRESHOLD
        for item in retrieved_evidence[1:]
    )

    if top_score < MODERATE_SIMILARITY_THRESHOLD:
        verdict = "No relevant evidence"
        evidence_status = "No close clinical evidence found in trusted repository"
        risk = "Medium"
    elif has_conflict:
        verdict = "Uncertain / Insufficient evidence"
        evidence_status = "Conflicting clinical evidence found among top sources"
        risk = "Medium"
    elif top_score >= HIGH_SIMILARITY_THRESHOLD:
        if top_stance == "refutes":
            verdict = "Refuted"
            evidence_status = "Directly refuted by reputable clinical evidence"
            claim_lower = claim.lower()
            is_dangerous = any(pat in claim_lower for pat in DANGEROUS_PATTERNS)
            risk = "Very High" if is_dangerous else "High"
        elif top_stance == "supports":
            verdict = "Supported"
            evidence_status = "Corroborated by verified medical sources"
            risk = "Low"
        else:
            verdict = "Uncertain / Insufficient evidence"
            evidence_status = "Evidence stance inconclusive"
            risk = "Medium"
    else:
        verdict = "Uncertain / Insufficient evidence"
        evidence_status = "Moderate similarity; insufficient clinical evidence"
        risk = "Medium"

    return {
        "verdict": verdict,
        "evidence_status": evidence_status,
        "risk": risk,
        "confidence": top_score,
        "stance": top_stance
    }""")

# Step 9
add_md("""## Step 9 — Standardized Educational Explanation Generator
Adheres to medical AI compliance requirements:
- Uses a fixed, transparent template rather than an unconstrained hallucination generator.
- Clearly states the verdict, retrieved evidence status, and risk level.
- Prominently embeds a clinical disclaimer that the system does not substitute for a medical doctor.""")

add_code("""def generate_explanation(original_claim, detected_language, is_health, verdict_data, retrieved_evidence):
    verdict = verdict_data["verdict"]
    risk = verdict_data["risk"]
    evidence_status = verdict_data["evidence_status"]

    if not is_health:
        summary_text = (
            "The claim was classified as **NOT health-related**. "
            "Our clinical verification engine only evaluates health, medical, or biomedical statements. "
            "No clinical retrieval was performed."
        )
        return {
            "original_claim": original_claim,
            "detected_language": detected_language,
            "is_health_related": False,
            "verdict": "Non-Health Claim",
            "evidence_status": "Not Applicable",
            "risk": "None",
            "summary_text": summary_text,
            "retrieved_evidence": [],
            "disclaimer": "This tool is for educational purposes only and does not constitute medical advice."
        }

    if verdict == "Supported":
        action_phrase = "supports"
    elif verdict == "Refuted":
        action_phrase = "refutes"
    else:
        action_phrase = "does not sufficiently address"

    summary_text = (
        f"The claim was classified as **{verdict}** based on the most relevant retrieved medical evidence.\\n\\n"
        f"The evidence **{action_phrase}** the claim.\\n\\n"
        f"- **Assessed Risk**: `{risk}`\\n"
        f"- **Evidence Status**: {evidence_status}\\n\\n"
        f"⚠️ *This tool is for educational purposes and is not medical advice.*"
    )

    return {
        "original_claim": original_claim,
        "detected_language": detected_language,
        "is_health_related": True,
        "verdict": verdict,
        "evidence_status": evidence_status,
        "risk": risk,
        "summary_text": summary_text,
        "retrieved_evidence": retrieved_evidence,
        "disclaimer": "This tool is for educational purposes and is not medical advice. Always consult a certified physician for medical concerns."
    }""")

# Step 10
add_md("""## Step 10 — Interactive Gradio Demo Interface
Launch an interactive web UI. The evaluator can input arbitrary claims or click on preset demo test claims:
1. `Haldi doodh cancer ko theek karta hai` (Hinglish -> Refuted -> High Risk)
2. `Vaccines autism cause karte hain` (Hinglish -> Refuted -> High Risk)
3. `Smoking causes lung cancer` (English -> Supported -> Low Risk)
4. `Antibiotics cold ko cure karte hain` (Hinglish -> Refuted -> High Risk)
5. `India won the cricket match` (English -> Non-Health -> Filtered)""")

add_code("""def verify_pipeline(claim: str):
    if not claim or not claim.strip():
        return "N/A", "N/A", "Please enter a claim.", "N/A", "N/A", "", ""

    cleaned = preprocess_text(claim)
    lang = detect_language(claim)
    is_health = is_health_related(cleaned)

    if not is_health:
        v_data = {"verdict": "Non-Health Claim", "evidence_status": "Filtered", "risk": "None", "confidence": 0.0, "stance": "none"}
        exp = generate_explanation(claim, lang, False, v_data, [])
        return lang, "No (Filtered Out)", "Non-Health Claim", "None", exp["summary_text"], "No clinical retrieval needed.", exp["disclaimer"]

    retrieved = retrieve_evidence(cleaned, top_k=3)
    v_data = assess_verdict_and_risk(cleaned, retrieved)
    exp = generate_explanation(claim, lang, True, v_data, retrieved)

    evidence_md = "### 📚 Retrieved Clinical Evidence\\n\\n"
    for item in retrieved:
        evidence_md += (
            f"**Rank {item['rank']}** [Cosine Similarity: `{item['similarity_score']}` | Stance: `{item['stance'].upper()}`]\\n"
            f"> *\"{item['evidence_text']}\"*\\n"
            f"**Source**: {item['source']} — [Verified URL Link]({item['url']})\\n\\n"
            f"---\\n"
        )

    return lang, "Yes (Passed Filter)", v_data["verdict"], v_data["risk"], exp["summary_text"], evidence_md, exp["disclaimer"]

with gr.Blocks(title="Healthcare Misinformation RAG") as demo:
    gr.Markdown("# 🩺 Multilingual Healthcare Misinformation Verifier (RAG)")
    gr.Markdown("Verify medical statements across **English**, **Hindi (हिंदी)**, and **Hinglish** using cross-lingual dense retrieval.")

    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(label="Enter Claim", placeholder="e.g. Haldi doodh cancer ko theek karta hai", lines=2)
            verify_button = gr.Button("🔍 Verify Claim", variant="primary")
            gr.Markdown("### Demo Test Inputs")
            gr.Examples(
                examples=[
                    ["Haldi doodh cancer ko theek karta hai"],
                    ["Vaccines autism cause karte hain"],
                    ["Smoking causes lung cancer"],
                    ["Antibiotics cold ko cure karte hain"],
                    ["India won the cricket match"]
                ],
                inputs=[input_box]
            )

        with gr.Column(scale=2):
            with gr.Row():
                out_lang = gr.Textbox(label="Detected Language", interactive=False)
                out_health = gr.Textbox(label="Health-Related?", interactive=False)
            with gr.Row():
                out_verdict = gr.Textbox(label="Scientific Verdict", interactive=False)
                out_risk = gr.Textbox(label="Risk Assessment", interactive=False)

    with gr.Row():
        with gr.Column():
            out_explanation = gr.Markdown(label="Explanation")
            out_evidence = gr.Markdown(label="Retrieved Evidence")
            out_disclaimer = gr.Markdown(label="Disclaimer")

    verify_button.click(
        fn=verify_pipeline,
        inputs=[input_box],
        outputs=[out_lang, out_health, out_verdict, out_risk, out_explanation, out_evidence, out_disclaimer]
    )

# Launch Gradio interface (share=True creates a public Colab URL)
demo.launch(share=True, debug=False)""")

target_path = "/Users/rishi/.gemini/antigravity/scratch/multilingual-healthcare-rag/Multilingual_Healthcare_Misinformation_RAG.ipynb"
with open(target_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"✅ Successfully wrote notebook to {target_path}")
