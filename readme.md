# 🏛️ Philosophy RAG Assistant

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](Dockerfile)

A production-grade, low-latency **Retrieval-Augmented Generation (RAG)** service delivering grounded, contextual philosophical synthesis based on vectorized literature.

Designed with an ultra-lightweight memory footprint (<400MB RAM), bypassing heavyweight CUDA/PyTorch dependencies in favor of ONNX Runtime CPU execution and sub-second Groq LLM inference.

---

## ⚡ Architectural Overview

The application follows an asynchronous, decoupled microservice pattern:

```text
[User Query] 
     │
     ▼
[Gradio UI / FastAPI REST API]
     │
     ▼
[ONNX Runtime / FastEmbed] ──> Dense Embedding Generation (384-d vector)
     │
     ▼
[Vector Store & Filter]   ──> Cosine Similarity Search + Metadata School Filter
     │
     ▼
[Augmented Payload]       ──> Top-K Literature Excerpts + Strict System Prompt
     │
     ▼
[Groq LLM Engine]         ──> Contextually Grounded Synthesis
```

### Key Engineering Decisions
- **Zero PyTorch Overhead:** Employs `fastembed` powered by `onnxruntime` (`sentence-transformers/all-MiniLM-L6-v2`), eliminating gigabytes of CUDA runtime dependencies and enabling fast deployment on resource-constrained infrastructure.
- **Guarded In-Memory Vector Search:** Precomputed NumPy embeddings paired with dynamic Pandas metadata filtering for sub-millisecond retrieval latencies.
- **Strict Anti-Hallucination Framing:** Prompt boundaries constrain LLM synthesis strictly to the retrieved contextual excerpts, preventing unsubstantiated extrapolation.
- **Isolated Serving Layers:** FastAPI exposes OpenAPI-compliant endpoints (`/recommend`, `/health`), while mounting an interactive Gradio UI (`/ui`) protected against pytest lifecycle conflicts.

---

## 🛠️ Tech Stack

| Layer | Technologies |
|---|---|
| **API & Serving** | FastAPI, Uvicorn, Pydantic v2, Gradio |
| **Retrieval & Embeddings** | FastEmbed, ONNX Runtime, NumPy, Scikit-learn |
| **Generation (LLM)** | Groq Cloud API (High-throughput open-weights inference) |
| **MLOps & CI** | Docker, GitHub Actions (Automated CI), Pytest, AnyIO |

---

## 📂 Project Structure

```text
PhilosophyBooks/
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow (Pytest & Docker build)
├── artifacts/
│   ├── embeddings.npy         # Precomputed normalized vector representations
│   ├── indexed_books.csv      # Corpus metadata (title, author, school, summary)
│   └── selected_model.txt     # Model registry reference
├── src/
│   ├── app/
│   │   └── main.py            # FastAPI service, Gradio UI & RAG pipeline
│   ├── indexing/
│   │   └── build_index.py     # Offline tokenization & vector indexing script
│   └── serving/
│       └── inference.py       # Core semantic retriever & cosine similarity engine
├── tests/
│   └── test_api.py            # Asynchronous integration & contract test suite
├── .env.example               # Template for required environment variables
├── .gitignore
├── Dockerfile                 # Multi-stage production container build
├── pytest.ini                 # Root test runner & pythonpath configuration
├── README.md
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Docker (optional, for containerized execution)
- A valid Groq Cloud API key ([console.groq.com](https://console.groq.com))

### 1. Local Setup

Clone the repository and install dependencies:
```bash
git clone [https://github.com/floreadavid2/PhilosophyBooks.git](https://github.com/floreadavid2/PhilosophyBooks.git)
cd PhilosophyBooks

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Configure your environment variables:
```bash
cp .env.example .env
# Edit .env and set your active key:
# GROQ_API_KEY=gsk_your_key_here
```

Launch the development server:
```bash
python -m uvicorn src.app.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Interactive Web UI:** [http://127.0.0.1:8000/ui](http://127.0.0.1:8000/ui)
- **Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🐳 Docker Deployment

The application is containerized for consistent production runtimes.

```bash
# Build the Docker image
docker build -t philosophy-rag:latest .

# Run container with environment configuration
docker run -d \
  -p 8000:8000 \
  --name philosophy-rag-app \
  --env-file .env \
  philosophy-rag:latest
```

---

## 🧪 Continuous Integration (CI) & Testing

The project uses **GitHub Actions** for Continuous Integration. Every push and pull request to `main` triggers an automated pipeline that:
1. Installs all production and test dependencies on a clean Ubuntu runner.
2. Executes asynchronous integration tests via `pytest` without invoking live LLM billing.
3. Compiles the production Docker image to ensure zero container build regressions.

To run the test suite locally:
```bash
pytest -v
```

---

## 📡 REST API Reference

### `POST /recommend`
Retrieves raw semantic matches without triggering the generative LLM layer.

**Request:**
```json
{
  "query": "coping with grief, suffering and events beyond our personal control",
  "top_k": 2,
  "school": "Stoicism"
}
```

**Response (200 OK):**
```json
{
  "query": "coping with grief, suffering and events beyond our personal control",
  "results_count": 2,
  "recommendations": [
    {
      "id": 4,
      "title": "Discourses and Selected Writings",
      "author": "Epictetus",
      "school": "Stoicism",
      "summary": "Core doctrines of Stoic discipline focusing on the dichotomy of control...",
      "similarity_score": 0.2841
    }
  ]
}
```

---

## 📄 License

Distributed under the MIT License.
