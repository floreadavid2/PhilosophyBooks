import time
from pathlib import Path
from fastembed import TextEmbedding
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/processed/philosophy_books_featured.csv")
ARTIFACTS_DIR = Path("artifacts")
EXPERIMENT_NAME = "philosophy-semantic-retrieval"

# Benchmark validation queries to test retrieval quality
BENCHMARK_QUERIES = [
    {
        "query": "defying the gods, creating own values, and the death of god",
        "expected_author": "Friedrich Nietzsche",
        "expected_school": "Nietzscheanism",
    },
    {
        "query": "dichotomy of control, emotional tranquility, and accepting fate",
        "expected_author": "Epictetus",
        "expected_school": "Stoicism",
    },
    {
        "query": "human nature is innately good with sprouts of virtue",
        "expected_author": "Mencius",
        "expected_school": "Confucianism",
    },
    {
        "query": "effortless action, simplicity, and living in flow with nature",
        "expected_author": "Lao Tzu",
        "expected_school": "Taoism",
    },
    {
        "query": "meaningless universe, absurdity, and living in continuous revolt",
        "expected_author": "Albert Camus",
        "expected_school": "Absurdism",
    },
]

# FastEmbed ONNX models: ultra-light, fast, no PyTorch DLL issues
CANDIDATE_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
]


def evaluate_retrieval(model: TextEmbedding, embeddings: np.ndarray, df: pd.DataFrame) -> dict:
    """Evaluate retrieval accuracy on curated benchmark queries."""
    top_1_correct = 0
    top_3_correct = 0

    for item in BENCHMARK_QUERIES:
        query_vec = np.array(list(model.embed([item["query"]])))
        similarities = cosine_similarity(query_vec, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:3]

        top_1_row = df.iloc[top_indices[0]]
        top_3_rows = df.iloc[top_indices]

        # Check Top-1 match by author or school
        if (top_1_row["author"] == item.get("expected_author")) or (
                top_1_row["school"] == item.get("expected_school")
        ):
            top_1_correct += 1

        # Check Top-3 match
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

    df = pd.read_csv(DATA_PATH)
    texts_to_embed = df["content_to_embed"].tolist()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)

    best_score = -1.0
    best_model_name = None

    print(f"Starting MLflow experiment: {EXPERIMENT_NAME}")

    for model_name in CANDIDATE_MODELS:
        run_name = model_name.split("/")[-1]
        print(f"\n--- Running evaluation for: {run_name} ---")

        with mlflow.start_run(run_name=run_name):
            # 1. Load ONNX-backed model
            model = TextEmbedding(model_name=model_name)

            # 2. Encode documents
            embed_start = time.perf_counter()
            embeddings_list = list(model.embed(texts_to_embed))
            embeddings = np.array(embeddings_list)
            index_latency = time.perf_counter() - embed_start

            model_dim = embeddings.shape[1]

            # 3. Evaluate retrieval performance
            metrics = evaluate_retrieval(model, embeddings, df)
            metrics["index_latency_seconds"] = index_latency
            metrics["embedding_dim"] = model_dim
            metrics["doc_count"] = len(df)

            # 4. Save model run artifacts locally
            safe_slug = run_name.replace("-", "_").replace(".", "_")
            embeddings_file = ARTIFACTS_DIR / f"embeddings_{safe_slug}.npy"
            np.save(embeddings_file, embeddings)

            # 5. Log parameters, metrics, and artifacts to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("backend", "onnxruntime")
            mlflow.log_param("distance_metric", "cosine")
            mlflow.log_param("embedding_dimension", model_dim)
            mlflow.log_metrics(metrics)

            mlflow.log_artifact(str(embeddings_file))
            mlflow.log_artifact(str(DATA_PATH))

            print(f"Metrics: {metrics}")

            # Keep track of the top-performing model
            score = metrics["top_1_accuracy"] + metrics["top_3_hit_rate"]
            if score > best_score:
                best_score = score
                best_model_name = model_name

                # Persist primary production artifacts
                np.save(ARTIFACTS_DIR / "embeddings.npy", embeddings)
                df.to_csv(ARTIFACTS_DIR / "indexed_books.csv", index=False)
                with open(
                        ARTIFACTS_DIR / "selected_model.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(best_model_name)

    print("\n=======================================================")
    print(f"Production model selected: {best_model_name}")
    print(f"Artifacts successfully saved into: {ARTIFACTS_DIR.resolve()}")
    print("=======================================================")


if __name__ == "__main__":
    run_indexing_pipeline()