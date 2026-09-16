"""
RAG Engine Module
Handles multilingual embeddings generation via SentenceTransformer and
dense vector search using FAISS.
"""

import os
import sys

# Prevent macOS OpenMP / Rust tokenizers thread conflict
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

try:
    from src.preprocessing import preprocess_text
except ModuleNotFoundError:
    from preprocessing import preprocess_text


class HealthcareRAGEngine:
    """
    RAG engine using paraphrase-multilingual-MiniLM-L12-v2 and FAISS IndexFlatIP
    for cross-lingual semantic matching between English/Hindi/Hinglish claims
    and trusted clinical evidence.
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        csv_path: str = None,
        df: pd.DataFrame = None,
    ):
        # Hardware auto-detection: GPU if available, else CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=self.device)

        # Load evidence dataframe
        if df is not None:
            self.df = df.copy()
        elif csv_path and os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
        else:
            raise ValueError("Either a valid DataFrame or an existing CSV path must be provided.")

        self.index = None
        self._build_index()

    def _build_index(self):
        """Encodes evidence texts and populates the FAISS inner-product index."""
        texts = self.df["evidence_text"].tolist()
        # Compute normalized embeddings for exact cosine similarity
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        )
        embeddings = np.array(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]
        # IndexFlatIP computes inner product; with normalized vectors, this is cosine similarity
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def retrieve_evidence(self, claim: str, top_k: int = 3):
        """
        Retrieves top_k closest evidence snippets for the given claim.
        Returns a list of dictionaries with evidence metadata and cosine similarity scores.
        """
        cleaned_claim = preprocess_text(claim)
        query_vector = self.model.encode(
            [cleaned_claim],
            normalize_embeddings=True,
            show_progress_bar=False,
            device=self.device
        )
        query_vector = np.array(query_vector, dtype=np.float32)

        scores, indices = self.index.search(query_vector, min(top_k, len(self.df)))

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            row = self.df.iloc[idx].to_dict()
            row["similarity_score"] = float(round(float(score), 4))
            row["rank"] = rank + 1
            results.append(row)

        return results
