from pathlib import Path
from typing import Any, Dict, List
from fastembed import TextEmbedding
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = Path("artifacts")
EMBEDDINGS_PATH = ARTIFACTS_DIR / "embeddings.npy"
DATA_PATH = ARTIFACTS_DIR / "indexed_books.csv"
MODEL_FILE_PATH = ARTIFACTS_DIR / "selected_model.txt"


class SemanticRecommender:

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR):
        self.artifacts_dir = artifacts_dir
        self.model = None
        self.embeddings = None
        self.df = None
        self.model_name = None
        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load precomputed embeddings, book metadata, and embedding model."""
        if not EMBEDDINGS_PATH.exists() or not DATA_PATH.exists():
            raise FileNotFoundError(
                f"Missing required artifacts in {self.artifacts_dir}. "
                "Run `python src/indexing/build_index.py` first."
            )

        # 1. Load selected model name
        if MODEL_FILE_PATH.exists():
            with open(MODEL_FILE_PATH, "r", encoding="utf-8") as f:
                self.model_name = f.read().strip()
        else:
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"

        # 2. Load model and vector index
        self.model = TextEmbedding(model_name=self.model_name)
        self.embeddings = np.load(EMBEDDINGS_PATH)
        self.df = pd.read_csv(DATA_PATH)

    def recommend(
            self, query: str, top_k: int = 3, school_filter: str = None
    ) -> List[Dict[str, Any]]:
        """Compute cosine similarity between query and books, returning top_k matches."""
        if not query or not query.strip():
            return []

        # Vectorize incoming query
        query_vec = np.array(list(self.model.embed([query.strip()])))

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, self.embeddings)[0]

        temp_df = self.df.copy()
        temp_df["similarity_score"] = similarities

        # Optional metadata filtering
        if school_filter and school_filter.lower() != "all":
            temp_df = temp_df[
                temp_df["school"].str.lower() == school_filter.lower()
                ]

        if temp_df.empty:
            return []

        # Rank and take top-k
        top_matches = temp_df.sort_values(
            by="similarity_score", ascending=False
        ).head(top_k)

        results = []
        for _, row in top_matches.iterrows():
            results.append(
                {
                    "id": int(row["id"]),
                    "title": str(row["title"]),
                    "author": str(row["author"]),
                    "school": str(row["school"]),
                    "summary": str(row["summary"]),
                    "similarity_score": round(float(row["similarity_score"]), 4),
                }
            )

        return results


# Global singleton instance for serving
recommender = SemanticRecommender()


def get_recommendations(
        query: str, top_k: int = 3, school: str = None
) -> List[Dict[str, Any]]:
    """Helper wrapper for API and UI layers."""
    return recommender.recommend(query=query, top_k=top_k, school_filter=school)