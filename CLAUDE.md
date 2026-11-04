## 1. Project Overview & Primary Tech Stack

This is a multi-agent clinical research integrity system engineered to detect outcome switching, safety reporting discrepancies, and publication delays. It achieves this by ingesting and reconciling study records from the ClinicalTrials.gov API and the PubMed eUtils API using a specialized routing workflow.

### Primary Tech Stack

* **Language:** Python 3.11.9

* **Web Framework:** FastAPI

* **Agent Orchestration:** LangGraph

* **Database & Vector Store:** PostgreSQL with `pgvector`

* **Storage & Deployment:** Google Cloud Storage (GCS), Google Cloud Run, Docker

* **Evaluation:** Layer-1 Unit Testing Each Agent in Isolation + LLM-as-a-judge scoring frameworks

## 2. Development, Build, and Test Commands

* **Install Dependencies:**

  `pip install -r requirements.txt` *(or via `pip install -e .` utilizing `pyproject.toml`)*

* **Run Dev Server:**

  `uvicorn api.main:app --reload`

* **Run Ingestion Pipeline:**

  `python ingestion/run_ingestion.py`

* **Run Processing Pipeline:**

  `python processing/run_processing.py`

* **Run Unit Tests:**

  `pytest tests/` *(for agent-specific isolation tests run `pytest tests/layer01_unit_tests_isolation/`)*

* **Linting & Formatting:**

  `black . && ruff check .`

## 3. Codebase Architecture & Core Entry Points

* `agents/`: Contains the central supervisor router (`supervisor.py`) alongside the 6 specialist agents: `broken_promises`, `missing_results`, `pattern_finder`, `side_effect`, `timeline`, and `track_record`.

* `api/`: The FastAPI backend. `api/main.py` is the primary application entry point. Routing logic is segregated into `analysis`, `memory`, `review`, and `signals`.

* `graph/`: Houses state management definitions (`state.py`) and the centralized LangGraph workflow builder (`graph_builder.py`), including human-in-the-loop review routing (`hitl.py`).

* `memory/`: Implementation of the three-layer memory model, distributing context across `episodic_store.py`, `procedural_store.py`, and `semantic_store.py`.

* `ingestion/` & `processing/`: Data infrastructure. `ingestion/` executes API requests and document parsing. `processing/` manages text chunking and embeds data into `pgvector`.

* `tests/`: Currently, only tests each agent in isolation using `pytest` and mocking libraries like `patch`, `MagicMock` and `AsyncMock`.

* `deployment/`: Environment configurations, GCP Cloud Run YAMLs, and Dockerfiles for deployment.

## 4. Code Style Guidelines, Patterns & Paradigms

* **Multi-Agent Supervisor Pattern:** Agents do not interact directly with one another. The architecture enforces a strict hub-and-spoke routing paradigm through `supervisor.py`, with each agent restricted to a tightly scoped domain.

* **Evaluation-Driven Development (EDD):** LLM performance and agent reasoning must be tested using the LLM-as-a-judge framework. While standard components use isolated unit tests (`layer01_unit_tests_isolation`), agent logic is continuously evaluated through `evaluation/judge.py` and `evaluation/scorers.py`.

* **Three-Layer Memory Separation:** State persistence explicitly isolates semantic (vector/embeddings), episodic (event history), and procedural (instructions/rules) memory types. Context retrieval paths must not cross-contaminate these stores.

* **Human-in-the-Loop (HITL) Enforcements:** High-stakes automated decisions must route through `graph/hitl.py` and the `api/routers/review.py` endpoints for validation rather than executing autonomously.

* **API Component Isolation:** FastAPI routes rely strictly on dependency injection (`api/dependencies.py`) and enforce rigorous typing and validation via Pydantic (`api/schemas.py`).

* **Tool Modularity:** External data queries and utilities are abstracted into shared toolsets (`tools/pubmed_tools.py`, `tools/clinical_tools.py`) rather than being hardcoded inside individual agent scripts.