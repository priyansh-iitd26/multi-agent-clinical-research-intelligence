# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development
- Install dependencies: `pip install -r requirements.txt && pip install -e .`
- Run API locally: `uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload`
- Explore API via Swagger UI: `http://localhost:8000/docs`
- Run tests: `pytest tests/`
- Run specific test file: `pytest tests/test_ingestion.py` or `pytest tests/test_processing.py`

### Data Pipeline
- Run Ingestion: `python3 ingestion/run_ingestion.py` (ClinicalTrials.gov/PubMed -> GCS)
- Run Processing: `python3 processing/run_processing.py` (Chunking/Embedding -> Cloud SQL)

### Deployment
- Deploy to GCP: `./deployment/gcp/deploy.sh`
- Database Initialization: Run `deployment/gcp/cloudsql_init.sql` against the Cloud SQL instance.

## Architecture & Structure

### High-Level Flow
The system is a multi-agent pipeline for detecting research integrity signals in clinical trials.

1. **Ingestion Layer** (`/ingestion`): Fetches raw trial data from ClinicalTrials.gov and PubMed; persists raw data in Google Cloud Storage (GCS).
2. **Processing Layer** (`/processing`): Implements a RAG pipeline (chunking $\rightarrow$ embedding $\rightarrow$ vector storage) using OpenAI `text-embedding-3-small` and PostgreSQL with `pgvector`.
3. **Memory Layer** (`/memory`): A unique three-tier storage system:
   - **Episodic**: Semantic search over past analysis sessions.
   - **Procedural**: Stores reasoning rules learned from human corrections (HITL).
   - **Semantic**: Maintains persistent sponsor credibility profiles.
4. **Agent Graph** (`/agents`, `/graph`): Built with LangGraph. A **Supervisor** coordinates 6 specialist agents (Missing Results, Broken Promises, Track Record, Pattern Finder, Side Effect Checker, Timeline Analyst) that execute in parallel.
5. **HITL Gate** (`/graph/hitl.py`): Signals are filtered by confidence. Low-confidence signals enter a review queue; human rejections trigger a feedback loop that updates the **Procedural Memory**, permanently altering how agents reason in future runs.
6. **API Layer** (`/api`): FastAPI endpoints for triggering analyses, managing the review queue, and inspecting memory.

### Key Design Patterns
- **Parallel Agent Execution**: Agents run concurrently rather than sequentially to handle large-scale study reasoning.
- **HITL Feedback Loop**: Human decisions directly update the agent's procedural "rules of reasoning" stored in the database.
- **Async IO**: Heavy use of `asyncio` and `asyncpg` for high-throughput data ingestion and API responses.
