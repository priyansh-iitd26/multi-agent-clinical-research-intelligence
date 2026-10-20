# Multi-Agent Clinical Trials Research Integrity System

> A production-grade multi-agent AI system that detects research integrity
> signals across clinical trials using 6 specialist agents running in parallel
> on Google Cloud Platform.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green.svg)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.6-orange.svg)](https://langchain-ai.github.io/langgraph)
[![GCP](<https://img.shields.io/badge/GCP-Cloud%20Run-blue.svg>)](https://cloud.google.com/run)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📖 Table of Contents

- [Project Description](#-project-description)
- [Demo](#-demo)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Environment Variables](#-environment-variables)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Deployment](#-deployment)
- [Troubleshooting](#-troubleshooting)
- [Roadmap](#-roadmap)
- [License](#-license)

---

## 📌 Project Description

This is a production-grade multi-agent AI system that ingests clinical trials data from ClinicalTrials.gov
and PubMed, reasons across studies simultaneously, detects research integrity
signals, and exposes a FastAPI layer - deployed on Google Cloud Run.

**The problem it solves:**
About 30% of completed clinical trials never publish their results. Outcome
switching, timeline delays, and safety discrepancies are buried across
400,000+ government records that no single human analyst can read simultaneously.
Our system reads the studies at once and surfaces what humans usually miss.

**What makes it different:**

- 6 specialist agents run **in parallel** - not sequentially
- **Three types of memory** — episodic, procedural, semantic
- **Learns from human feedback** — rejections update agents reasoning permanently
- **Production deployed** on GCP

---

## 🎬 Demo (to be updated)

```bash
# Run a live analysis against production
curl -s -X POST \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{"task": "Find completed clinical trials where results were never posted", "max_studies": 3}' \
  https://mosaic-api-569957100480.us-central1.run.app/api/v1/analyze | python3 -m json.tool
```

**Sample response:**

```json
{
  "run_id": "a5712262-7b94-4bfc-9d46-6861000681fe",
  "task": "Find completed clinical trials where results were never posted",
  "final_brief": "Three completed clinical trials identified with results overdue by several years...",
  "total_signals": 3,
  "signals_requiring_review": 0,
  "agents_activated": ["missing_results_agent", "track_record_agent"],
  "duration_seconds": 15.24
}
```

---

## ✨ Features

### 6 Specialist Agents Running in Parallel

| Agent               | What it detects                                                           |
| ------------------- | ------------------------------------------------------------------------- |
| Missing Results     | Completed trials with no results posted (legal violation after 12 months) |
| Broken Promises     | Outcome switching - goals changed mid-study                              |
| Track Record        | Sponsor credibility scores built over time                                |
| Pattern Finder      | Cross-study patterns invisible to single-study readers                    |
| Side Effect Checker | Safety gaps between official filings and published papers                 |
| Timeline Analyst    | Silent delays past completion date with no explanation                    |

### Three-Layer Memory System

- **Episodic** - agents remember what they found in past sessions
- **Procedural** - agents learn from human corrections permanently
- **Semantic** - sponsor knowledge base grows with every analysis run

### Human-in-the-Loop (HITL) Gate

- Low confidence signals → human review queue
- Rejections → written to procedural memory → agent reasons differently forever
- One correction changes agent behaviour in ALL future sessions

### Production Infrastructure

- Serverless on Cloud Run - scales to zero, costs nothing when idle
- PostgreSQL + pgvector - semantic search over 1536-dimensional embeddings
- 9 endpoints with FastAPI + auto-generated Swagger docs

---

## 🛠 Tech Stack

| Layer           | Technology                                     |
| --------------- | ---------------------------------------------- |
| Language        | Python 3.12 async                              |
| Agent Framework | LangGraph 0.2.6                                |
| Reasoning ChatModel | meta-llama/Llama-3.3-70B-Instruct          |
| Embeddings      | Qwen/Qwen3-Embedding-8B (1536 dims)            |
| API             | FastAPI                                        |
| Database        | PostgreSQL 15 + pgvector (Cloud SQL)           |
| Vector Search   | pgvector HNSW index                            |
| File Storage    | Google Cloud Storage                           |
| Deployment      | Google Cloud Run                               |
| Secrets         | GCP Secret Manager                             |
| HTTP Clients    | requests (ClinicalTrials.gov) + httpx (PubMed) |
| Retry Logic     | tenacity                                       |
| Validation      | Pydantic v2                                    |

---

# 🏗️ Architecture

## 1. DATA SOURCES

- **ClinicalTrials.gov API v2** — 400K+ studies, free, no authentication required.
- **PubMed eUtils API** — Research papers, free, no authentication required.

↓

## 2. INGESTION LAYER

- **`clinical_trials_client.py`** — Asynchronous requests, rate limiting, and retries.
- **`pubmed_client.py`** — ESearch → EFetch, followed by XML parsing.
- **`document_parser.py`** — Converts raw JSON into Pydantic models.
- **`gcs_store.py`** — Saves raw and processed data to Google Cloud Storage (GCS).

↓

## 3. PROCESSING LAYER

- **`chunker.py`** — Splits documents into 500-word chunks with a 50-word overlap.
- **`embedder.py`** — Generates 1,536-dimensional embeddings in batches of 50.
- **`vector_store.py`** — Uses `asyncpg` and `pgvector` for cosine-similarity search.

↓

## 4. MEMORY LAYER

- **`episodic_store.py`** — Stores past sessions and supports semantic retrieval.
- **`procedural_store.py`** — Stores reasoning rules and supports human-in-the-loop (HITL) learning.
- **`semantic_store.py`** — Maintains sponsor credibility profiles.

↓

## 5. AGENT GRAPH

**Framework:** LangGraph  
**Execution:** Parallel execution of specialist agents, coordinated by the supervisor.

1. **Supervisor Agent** — Routes tasks and compiles the final brief.
2. **Missing Results Agent** — Confidence threshold: `0.65`.
3. **Broken Promises Agent** — Confidence threshold: `0.60`.
4. **Track Record Agent** — Confidence threshold: `0.70`.
5. **Pattern Finder Agent** — Confidence threshold: `0.65`.
6. **Side Effect Checker** — Confidence threshold: `0.55`.
7. **Timeline Analyst** — Confidence threshold: `0.60`.

↓

## 6. HUMAN-IN-THE-LOOP (HITL)

- **High-confidence findings** — Saved directly to the `signals` table.
- **Low-confidence findings** — Sent to a review queue for a human decision; the feedback updates procedural memory.

↓

## 7. API & DEPLOYMENT

- **FastAPI** — Exposes 9 endpoints.
- **Deployment platform** — Google Cloud Run.

## 📁 Project Structure

```text
work-dir/
├── config/
│   ├── settings.py                 # Pydantic BaseSettings, all environment variables
│   └── logging_config.py           # Centralised logging setup
│
├── ingestion/
│   ├── clinical_trials_client.py
│   ├── pubmed_client.py
│   ├── document_parser.py
│   ├── gcs_store.py
│   └── run_ingestion.py            # entry point — run this first
│
├── processing/
│   ├── chunker.py
│   ├── embedder.py
│   ├── vector_store.py
│   └── run_processing.py           # entry point — run this second
│
├── memory/
│   ├── episodic_store.py
│   ├── procedural_store.py
│   └── semantic_store.py
│
├── agents/
│   ├── supervisor.py
│   ├── broken_promises_agent.py
│   ├── missing_results_agent.py
│   ├── track_record_agent.py
│   ├── pattern_finder_agent.py
│   ├── side_effect_agent.py
│   └── timeline_agent.py
│
├── tools/
│   ├── search_tools.py              # 8 database + memory tools
│   ├── clinical_tools.py            # 4 live ClinicalTrials.gov tools
│   └── pubmed_tools.py              # 3 live PubMed tools
│
├── graph/
│   ├── state.py                     # GraphState TypedDict
│   ├── graph_builder.py             # wires all agents into LangGraph
│   └── hitl.py                      # HITL gate + learning loop
│
├── api/
│   ├── main.py                      # FastAPI app + lifespan
│   ├── schemas.py                   # All Pydantic request/response models
│   ├── dependencies.py              # Shared resource singletons
│   └── routers/
│       ├── analysis.py              # POST /api/v1/analyze
│       ├── signals.py               # GET /api/v1/signals
│       ├── review.py                # GET + PATCH /api/v1/review
│       └── memory.py                # GET /api/v1/memory + /sponsors
│
├── deployment/
│   ├── Dockerfile                   
│   ├── .dockerignore
│   └── gcp/
│       ├── cloudsql_init.sql        # schema - run once after Cloud SQL setup
│       └── deploy.sh                # full GCP deployment script
│
├── .gitignore
├── pyproject.toml                
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Docker Desktop (for deployment)
- Google Cloud account with billing enabled
- HuggingFace Access Token

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/mosaic.git
cd mosaic
```

### 2. Create virtual environment

```bash
python3 -m venv .venv
.venv\Scripts\activate  # On Mac: source .venv/bin/activate 
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Set up GCP infrastructure

```bash
# Authenticate
gcloud auth login
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Enable APIs
gcloud services enable sqladmin.googleapis.com storage.googleapis.com run.googleapis.com secretmanager.googleapis.com --project=YOUR_PROJECT_ID

# Create Cloud Storage bucket
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://YOUR_BUCKET_NAME

# Create Cloud SQL instance
gcloud sql instances create clinical-trial-db \
  --database-version=POSTGRES_15 \
  --tier=db-f1-micro \
  --zone=us-central1-f \
  --project=YOUR_PROJECT_ID

# Create database and user
gcloud sql databases create clinical_trial_db --instance=clinical-trial-db
gcloud sql users create mosaic_user --instance=clinical-trial-db --password=YOUR_PASSWORD
```

### 5. Create the database schema

```bash
# Whitelist your IP first
curl -4 ifconfig.me  # copy this IP

gcloud sql instances patch clinical-trial-db \
  --authorized-networks=YOUR_IP/32 \
  --project=YOUR_PROJECT_ID

# Connect and run schema
psql "host=YOUR_SQL_IP port=5432 dbname=clinical_trial_db user=mosaic_user"
```

Then paste the contents of `deployment/gcp/cloudsql_init.sql`.

### 6. Configure environment variables

```bash
cp .env.example .env
# Fill in all values — see Environment Variables section below
```

### 7. Run ingestion pipeline

```bash
python3 ingestion/run_ingestion.py
# Downloads studies from ClinicalTrials.gov and PubMed → saves to GCS
```

### 8. Run processing pipeline

```bash
python3 processing/run_processing.py
# Chunks studies, generates embeddings, saves to Cloud SQL
```

### 9. Run tests (Optional)

```bash
pytest tests/
```

### 10. Run the API locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Open http://localhost:8000/docs for Swagger UI
```

---

## 🔐 Environment Variables

```env
# HuggingFace
HUGGINGFACE_API_KEY=your-huggingface-api-key

# GCP
GCP_PROJECT_ID=your-project-id
GCP_REGION=us-central1
GCS_BUCKET_NAME=your-bucket-name

# Cloud SQL
DB_HOST=34.XXX.XXX.XXX          # public IP for local dev
DB_PORT=5432
DB_NAME=clinical_trial_db
DB_USER=mosaic_user
DB_PASSWORD=your-password

# API
API_HOST=0.0.0.0
API_PORT=8000
API_ENV=development
```

> **Never commit `.env` to git.** It is in `.gitignore` by default.

---

## 📡 Usage

The system is operated primarily through the FastAPI API.

### 🧪 Local Testing & Exploration
If you are running the system locally, you can use the built-in Swagger UI:
`http://localhost:8000/docs`

### 🔍 Running an Analysis
Trigger the multi-agent graph to research a specific integrity signal:
```bash
curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"task": "Find completed trials with missing results", "max_studies": 5}' \
  http://localhost:8000/api/v1/analyze | python3 -m json.tool
```

### ⚖️ Human-in-the-Loop (HITL) Workflow
Low-confidence signals are routed to the review queue for expert validation.

**1. Check the review queue:**
```bash
curl -s http://localhost:8000/api/v1/review/queue | python3 -m json.tool
```

**2. Submit a review decision:**
Decision can be `approve`, `reject`, or `edit`. Rejections update the **Procedural Memory**, changing how agents reason in the future.
```bash
curl -s -X PATCH \
  -H "Content-Type: application/json" \
  -d '{"decision": "reject", "reviewer": "analyst@company.com", "rejection_reason": "Trial was terminated early — exempt from posting requirement"}' \
  http://localhost:8000/api/v1/review/QUEUE_ID_HERE | python3 -m json.tool
```

### 🧠 Memory Inspection
Inspect what the agents have learned over time.

**Search episodic memory (past sessions):**
```bash
curl -s "http://localhost:8000/api/v1/memory/episodes?query=missing+results+sponsor" | python3 -m json.tool
```

**View agent-specific reasoning rules (procedural memory):**
```bash
curl -s http://localhost:8000/api/v1/memory/procedures/missing_results_agent | python3 -m json.tool
```

---

## 📚 API Documentation

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/v1/analyze` | Trigger full multi-agent analysis run |
| `GET` | `/api/v1/signals` | List all detected integrity signals |
| `GET` | `/api/v1/signals/{id}` | Get detailed evidence for one signal |
| `GET` | `/api/v1/review/queue` | Fetch pending signals for human review |
| `PATCH` | `/api/v1/review/{id}` | Submit human decision (updates procedural memory) |
| `GET` | `/api/v1/memory/episodes` | Semantic search over past analysis sessions |
| `GET` | `/api/v1/memory/procedures/{agent}` | Retrieve current reasoning rules for a specific agent |
| `GET` | `/api/v1/sponsors` | List all tracked sponsor credibility profiles |
| `GET` | `/api/v1/sponsors/{name}` | Get a specific sponsor's integrity profile |
| `GET` | `/api/v1/health` | System health & dependency check |

---

## ☁️ Deployment

### Deploy to Google Cloud Run
The system is designed for serverless deployment on GCP to minimize costs.

```bash
# Make script executable (first time only)
chmod +x deployment/gcp/deploy.sh

# Run full deployment (Builds image -> Pushes to GCR -> Deploys to Cloud Run)
./deployment/gcp/deploy.sh
```

**Manual Deployment Steps:**
1. **Build & Push**: Use `docker buildx build --platform linux/amd64` to create an image compatible with Cloud Run.
2. **Deploy**: Use `gcloud run deploy` with 2Gi Memory and 2 CPUs to handle the parallel agent graph execution.

### Cost Optimization
To avoid ongoing Cloud SQL costs during development:
```bash
gcloud sql instances patch clinical-trial-db \
  --activation-policy=NEVER \
  --project=YOUR_PROJECT_ID
```

---

## 🗺 Roadmap
- [ ] **Evaluation Layer**: Integrate LangSmith for systematic agent performance evals.
- [ ] **Visual Dashboard**: Build a Streamlit interface for the HITL review queue.
- [ ] **Expanded Data**: Integrate FDA adverse event databases as a third primary source.
- [ ] **Automation**: Implement Cloud Scheduler for weekly integrity sweeps.
- [ ] **Alerting**: Email/Slack notifications for high-confidence critical signals.
- [ ] **Comparison**: Cross-sponsor integrity benchmarking.
- [ ] **Global Reach**: Support for EudraCT and ISRCTN registries.
- [ ] **Domain Tuning**: Fine-tune embedding model specifically for clinical trial terminology.

---

## 📄 License
MIT License — see [LICENSE](LICENSE) for details.