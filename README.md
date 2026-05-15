# 📚 LocalNotebook — RAG System with MLOps

> A local, privacy-first NotebookLM alternative built with production-grade NLP and MLOps practices.  
> Supports multi-source ingestion, semantic retrieval, and grounded generation with full experiment tracking.

---

## 🧠 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         LocalNotebook                           │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │  Sources  │───▶│  Ingestion   │───▶│  Vector Store      │    │
│  │ PDF/TXT  │    │  + Chunking  │    │  (ChromaDB)        │    │
│  │ HTML/MD  │    └──────────────┘    └────────┬───────────┘    │
│  └──────────┘                                 │                 │
│                                               ▼                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │  Answer  │◀───│  Generation  │◀───│  Retrieval         │    │
│  │ + Sources│    │  LLM (local) │    │  + Reranking       │    │
│  └──────────┘    └──────────────┘    └────────────────────┘    │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    MLOps Layer                            │  │
│  │   MLflow Tracking │ DVC Versioning │ RAGAS Evaluation    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer            | Technology                              | Role                                      |
|------------------|-----------------------------------------|-------------------------------------------|
| **LLM**          | Ollama + LFM2.5-1.2B / llama3:8b   | Local generation, no cloud dependency     |
| **Embeddings**   | `sentence-transformers` (multilingual-e5) | Semantic vector representation           |
| **Vector Store** | ChromaDB                                | Persistent local similarity search        |
| **Reranking**    | `cross-encoder/ms-marco-MiniLM`         | Precision improvement on top-k results    |
| **Ingestion**    | LangChain + Unstructured                | Multi-format document parsing             |
| **API**          | FastAPI                                 | Async REST endpoints, auto-docs           |
| **UI**           | Streamlit                               | Interactive notebook interface            |
| **Tracking**     | MLflow                                  | Experiment logging, metrics, artifacts    |
| **Versioning**   | DVC                                     | Data and pipeline reproducibility         |
| **Evaluation**   | RAGAS                                   | RAG-specific quality metrics              |
| **Containers**   | Docker + Docker Compose                 | One-command full stack deployment         |

---

## 🔬 Model Comparison Experiment

A core objective of this project is the **comparative evaluation** of two architecturally distinct LLMs for RAG:

| Model             | Architecture       | Params | RAM Usage | Notes                          |
|-------------------|--------------------|--------|-----------|-------------------------------|
| **LFM2.5-1.2B**  | Hybrid SSM+Attn    | 1.2B   | ~2 GB     | Edge-optimized, non-Transformer |
| **llama3:8b** | Transformer (GQA)  | 8B     | ~6 GB     | General-purpose baseline       |

Both models are evaluated on the same RAGAS metrics and tracked in MLflow for reproducible comparison.

---

## 📊 Evaluation Metrics (RAGAS)

| Metric                | Description                                              |
|-----------------------|----------------------------------------------------------|
| **Faithfulness**      | Is the answer grounded in the retrieved context?        |
| **Answer Relevancy**  | Does the answer address the question asked?             |
| **Context Precision** | Are the retrieved chunks actually relevant?             |
| **Context Recall**    | Were all relevant chunks successfully retrieved?        |

All metrics are logged per run in MLflow with model name, chunk size, top-k, and embedding model as parameters.

---

## 🗂️ Project Structure

```
localnotebook/
│
├── data/
│   ├── raw/                    # Original source documents (DVC tracked)
│   ├── processed/              # Generated chunks
│   └── eval/                   # Evaluation datasets (question/answer pairs)
│
├── src/
│   ├── ingestion/
│   │   ├── loader.py           # Multi-format document loading
│   │   └── chunker.py          # Chunking strategies (recursive, semantic)
│   ├── embeddings/
│   │   ├── encoder.py          # sentence-transformers wrapper
│   │   └── vectorstore.py      # ChromaDB interface
│   ├── retrieval/
│   │   ├── retriever.py        # Similarity search (top-k)
│   │   └── reranker.py         # Cross-encoder reranking
│   ├── generation/
│   │   └── llm.py              # Ollama client abstraction
│   ├── pipeline/
│   │   └── rag.py              # End-to-end RAG orchestration
│   └── evaluation/
│       └── ragas_eval.py       # RAGAS evaluation runner
│
├── api/
│   ├── main.py                 # FastAPI app + endpoints
│   └── schemas.py              # Pydantic request/response models
│
├── ui/
│   └── app.py                  # Streamlit interface
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_experiments.ipynb
│   └── 03_evaluation_analysis.ipynb
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_retrieval.py
│   └── test_pipeline.py
│
├── configs/
│   ├── config.yaml             # Global config (chunk_size, top_k, etc.)
│   └── models.yaml             # Model registry (LFM2.5 vs Llama3)
│
├── scripts/
│   ├── ingest.py               # CLI: ingest a source
│   └── evaluate.py             # CLI: run RAGAS evaluation
│
├── .github/
│   └── workflows/
│       └── ci.yml              # CI pipeline (lint, tests)
│
├── docker-compose.yml          # Full stack: API + UI + MLflow + ChromaDB
├── Dockerfile
├── Makefile                    # Shortcuts: make ingest, make eval, make up
├── dvc.yaml                    # Reproducible pipeline stages
├── params.yaml                 # DVC pipeline parameters
├── requirements.txt
├── requirements-dev.txt        # pytest, ruff, black
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Required
docker & docker-compose
ollama          # https://ollama.com
python 3.11+
```

### 1. Clone & configure

```bash
git clone https://github.com/yourname/localnotebook.git
cd localnotebook
cp .env.example .env
```

### 2. Pull LLM models

```bash
ollama pull hf.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF
ollama pull llama3.1:8b
```

### 3. Start the stack (Local)

```bash
# Easy local run (FastAPI + Streamlit + MLflow)
./scripts/run_local.sh
```

### 4. Ingest and Chat

1. Open Streamlit: [http://localhost:8501](http://localhost:8501)
2. Use the **Ingest** button in the sidebar.
3. Chat with your documents.
4. Rate answers with 👍/👎 to collect human feedback.
5. Click **End & Evaluate** to trigger RAGAS and sync everything to MLflow.

### 5. Access the interfaces

| Service      | URL                        |
|--------------|----------------------------|
| Streamlit UI | http://localhost:8501       |
| FastAPI docs | http://localhost:8000/docs  |
| MLflow UI    | http://localhost:5000       |

---

## 🔄 MLOps Pipeline (DVC)

```bash
# Run the full reproducible pipeline
dvc repro

# Run only evaluation
dvc repro evaluate

# Compare experiment runs
mlflow ui
```

Pipeline stages defined in `dvc.yaml`:
1. `ingest` — parse and chunk documents
2. `embed` — generate and store embeddings
3. `evaluate` — run RAGAS on both models

---

## 🧪 Running Tests

```bash
make test
# or
pytest tests/ -v --cov=src
```

---

## 📈 Experiment Tracking

Each RAG run logs to MLflow:

**Parameters tracked:**
- `model_name` (lfm2.5 | llama3.1)
- `embedding_model`
- `chunk_size`, `chunk_overlap`
- `top_k`, `reranking` (bool)

**Metrics tracked:**
- `faithfulness`, `answer_relevancy`
- `context_precision`, `context_recall`
- `latency_ms`, `tokens_per_second`

---
