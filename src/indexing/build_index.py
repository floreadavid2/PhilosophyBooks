import os
import sys

if sys.platform == "win32":
    site_packages = os.path.join(sys.prefix, "Lib", "site-packages")
    cuda_dirs = [
        os.path.join(site_packages, "nvidia", "cuda_runtime", "bin"),
        os.path.join(site_packages, "nvidia", "cublas", "bin"),
        os.path.join(site_packages, "nvidia", "cudnn", "bin"),
        os.path.join(site_packages, "nvidia", "cufft", "bin"),
    ]
    for d in cuda_dirs:
        if os.path.isdir(d):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]

import time
from pathlib import Path
from fastembed import TextEmbedding
import mlflow
import numpy as np
import pandas as pd

DATA_PATH = Path("artifacts/processed/indexed_books.csv")
ARTIFACTS_DIR = Path("artifacts/embeddings")
EXPERIMENT_NAME = "philosophy-semantic-retrieval"

BENCHMARK_QUERIES = [
    {
        "query": "defying the gods, creating own values, and the death of god",
        "expected_author": "Friedrich Nietzsche",
        "expected_school": "Existentialism / Nihilism",
    },
]

CANDIDATE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]

def calculate_cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Calculates cosine similarity strictly using numpy to avoid scipy/sklearn DLL issues."""
    query_vec = query_vec.flatten()
    dot_product = np.dot(doc_matrix, query_vec)
    norm_query = np.linalg.norm(query_vec)
    norm_docs = np.linalg.norm(doc_matrix, axis=1)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        similarity = np.where(norm_docs == 0, 0, dot_product / (norm_query * norm_docs))

    return similarity

def evaluate_retrieval(model: TextEmbedding, embeddings: np.ndarray, df: pd.DataFrame) -> dict:
    """Evaluate retrieval accuracy on curated benchmark queries."""
    if not BENCHMARK_QUERIES:
        return {"top_1_accuracy": 0.0, "top_3_hit_rate": 0.0}

    top_1_correct = 0
    top_3_correct = 0

    for item in BENCHMARK_QUERIES:
        query_vec = np.array(list(model.embed([item["query"]])))
        similarities = calculate_cosine_similarity(query_vec, embeddings)
        top_indices = np.argsort(similarities)[::-1][:3]

        top_1_row = df.iloc[top_indices[0]]
        top_3_rows = df.iloc[top_indices]

        if (top_1_row["author"] == item.get("expected_author")) or (
                top_1_row["school"] == item.get("expected_school")
        ):
            top_1_correct += 1

        top_3_authors = top_3_rows["author"].tolist()
        top_3_schools = top_3_rows["school"].tolist()
        if (item.get("expected_author") in top_3_authors) or (
                item.get("expected_school") in top_3_schools
        ):
            top_3_correct += 1

    total = len(BENCHMARK_QUERIES)
    return {
        "top_1_accuracy": top_1_correct / total,
        "top_3_hit_rate": top_3_correct / total,
    }

def run_indexing_pipeline():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing input dataset at {DATA_PATH}")

    print(f"[*] Loading dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    texts_to_embed = df["content_to_embed"].tolist()
    total_chunks = len(texts_to_embed)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.set_experiment(EXPERIMENT_NAME)

    best_score = -1.0
    best_model_name = None

    print(f"[*] Starting MLflow experiment: {EXPERIMENT_NAME}")
    print(f"[*] Total chunks to process: {total_chunks}\n")

    for model_name in CANDIDATE_MODELS:
        run_name = model_name.split("/")[-1]
        print(f"--- Running evaluation for: {run_name} ---")

        with mlflow.start_run(run_name=run_name):
            print(f"    -> Loading ONNX model '{run_name}' into memory...")
            model = TextEmbedding(model_name=model_name, providers=["CUDAExecutionProvider"])
            print(f"    -> Generating embeddings... This will take a few minutes.")
            embed_start = time.perf_counter()

            embeddings_list = []
            for i, embedding in enumerate(model.embed(texts_to_embed, batch_size=256)):
                embeddings_list.append(embedding)
                if (i + 1) % 500 == 0:
                    print(f"       ... embedded {i + 1}/{total_chunks} chunks ...")

            embeddings = np.array(embeddings_list)
            index_latency = time.perf_counter() - embed_start
            print(f"    -> Embedding completed in {index_latency:.2f} seconds.")

            model_dim = embeddings.shape[1]

            print(f"    -> Evaluating retrieval metrics...")
            metrics = evaluate_retrieval(model, embeddings, df)
            metrics["index_latency_seconds"] = index_latency
            metrics["embedding_dim"] = model_dim
            metrics["doc_count"] = total_chunks

            safe_slug = run_name.replace("-", "_").replace(".", "_")
            embeddings_file = ARTIFACTS_DIR / f"embeddings_{safe_slug}.npy"
            np.save(embeddings_file, embeddings)

            mlflow.log_param("model_name", model_name)
            mlflow.log_param("backend", "onnxruntime")
            mlflow.log_param("distance_metric", "cosine")
            mlflow.log_param("embedding_dimension", model_dim)
            mlflow.log_metrics(metrics)

            mlflow.log_artifact(str(embeddings_file))
            mlflow.log_artifact(str(DATA_PATH))

            print(f"    -> Metrics: {metrics}\n")

            score = metrics["top_1_accuracy"] + metrics["top_3_hit_rate"]
            if score > best_score:
                best_score = score
                best_model_name = model_name

                np.save(ARTIFACTS_DIR.parent / "embeddings.npy", embeddings)
                df.to_csv(ARTIFACTS_DIR.parent / "indexed_books.csv", index=False)
                with open(
                        ARTIFACTS_DIR.parent / "selected_model.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(best_model_name)

    print("=======================================================")
    print(f"[*] Production model selected: {best_model_name}")
    print(f"[*] Artifacts successfully saved into: {ARTIFACTS_DIR.parent.resolve()}")
    print("=======================================================")

if __name__ == "__main__":
    run_indexing_pipeline()