# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development
- Install dependencies: `pip install -r requirements.txt && pip install -e .`
- Run API locally: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`
- Access API Docs: `http://localhost:8000/docs`

### Data Pipeline
- Run Ingestion: `python3 ingestion/run_ingestion.py` (ClinicalTrials.gov/PubMed -> GCS)
- Run Processing: `python3 processing/run_processing.py` (Chunking/Embedding -> Cloud SQL)

### Deployment
- Deploy to GCP: `./deployment/gcp/deploy.sh`
- Init Database: Run `deployment/gcp/cloudsql_init.sql` against the Cloud SQL instance.

## Architecture & Structure

### High-Level Flow
1. **Ingestion** (`/ingestion`): Fetches raw trial data from ClinicalTrials.gov and PubMed; stores in GCS.
2. **Processing** (`/processing`): Chunks documents and generates 1,536-dim embeddings; stores in PostgreSQL with `pgvector`.
3. **Memory** (`/memory`): Three-layer system:
   - **Episodic**: Past session history.
   - **Procedural**: Human-in-the-loop (HITL) learned reasoning rules.
   - **Semantic**: Sponsor credibility profiles.
4. **Agent Graph** (`/agents`, `/graph`): A LangGraph-powered multi-agent system. A **Supervisor** routes tasks to 6 parallel specialist agents (Missing Results, Broken Promises, Track Record, Pattern Finder, Side Effect Checker, Timeline Analyst).
5. **HITL Gate** (`/graph/hitl.py`): High-confidence signals are saved; low-confidence signals go to a review queue. Human rejections update Procedural memory.
6. **API** (`/api`): FastAPI layer exposing analysis, signal review, and memory endpoints.

### Key Directories
- `config/`: Environment settings and logging.
- `tools/`: Shared tools for DB search, ClinicalTrials.gov, and PubMed APIs.
- `deployment/`: Docker and GCP infrastructure scripts.
