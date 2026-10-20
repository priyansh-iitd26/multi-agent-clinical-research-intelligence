CREATE TABLE IF NOT EXISTS studies (
    nct_id              TEXT PRIMARY KEY,
    title               TEXT,
    sponsor             TEXT,
    phase               TEXT,
    status              TEXT,
    conditions          TEXT[],
    interventions       TEXT[],
    primary_outcome     TEXT,
    secondary_outcomes  TEXT[],
    start_date          TEXT,
    completion_date     TEXT,
    results_posted      BOOLEAN,
    enrollment          INT,
    gcs_path            TEXT,
    ingested_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nct_id          TEXT REFERENCES studies(nct_id),
    chunk_text      TEXT,
    embedding       VECTOR(1536),
    chunk_index     INT,
    source          TEXT
);

CREATE TABLE IF NOT EXISTS signals (
    signal_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    nct_id          TEXT REFERENCES studies(nct_id),
    agent           TEXT,
    signal_type     TEXT,
    summary         TEXT,
    evidence        JSONB,
    confidence      FLOAT,
    status          TEXT DEFAULT 'pending',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS hitl_reviews (
    review_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    signal_id           UUID REFERENCES signals(signal_id),
    reviewer            TEXT,
    decision            TEXT,
    edit_summary        TEXT,
    rejection_reason    TEXT,
    reviewed_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sponsor_profiles (
    sponsor             TEXT PRIMARY KEY,
    credibility_score   FLOAT,
    total_studies       INT,
    results_posted      INT,
    avg_delay_days      FLOAT,
    broken_promises     INT,
    last_updated        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- creating the episodes table

CREATE TABLE IF NOT EXISTS episodes(
  episode_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_name TEXT NOT NULL,
  nct_id TEXT,
  content TEXT NOT NULL,
  outcome TEXT,
  embedding VECTOR(1536),
  created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS episodes_embedding_idx
  ON episodes USING hnsw (embedding vector_cosine_ops);

-- creating the procedures table

CREATE TABLE IF NOT EXISTS procedures (
  procedure_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_name TEXT NOT NULL,
  rule_text TEXT NOT NULL,
  rule_type TEXT DEFAULT 'learned',
  source TEXT DEFAULT 'default',
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS procedures_agent_idx
  ON procedures(agent_name);

### The Five Tables — Plain Language Explanation

---

#### Where do all these tables live?

All five tables live inside:

* **Cloud SQL instance** : `clinical-trial-db` (the server)
* **Database** : `clinical_trial_db` (the storage space inside the server)
* **Location** : Google Cloud, `us-central1-a`, running 24/7

Think of it like this:

```
Google Cloud
    └── Cloud SQL instance (clinical-trial-db)  ← the server
            └── Database (clinical_trial_db)     ← the storage space
                    ├── studies table
                    ├── chunks table
                    ├── signals table
                    ├── hitl_reviews table
                    └── sponsor_profiles table
```

---

#### Table 1 — `studies`

**What it stores:**
One row per clinical trial. Every study we download from ClinicalTrials.gov gets one row here.

**Example row:**

```
nct_id          : NCT04788680
title           : Semaglutide for Type 2 Diabetes
sponsor         : Novo Nordisk
phase           : PHASE3
status          : COMPLETED
conditions      : ["Type 2 Diabetes", "Obesity"]
primary_outcome : Reduction in HbA1c at 26 weeks
results_posted  : false
enrollment      : 1200
```

**Why we need it:**
This is the master record of every study in our system. Every other table references this table. When an agent generates a signal, it links back to a row in this table via `nct_id`. When we do a semantic search, we join results back to this table to get the full study context.

**Who writes to it:**
`gcs_store.py` during ingestion, and `vector_store.py` during processing.

**Who reads from it:**
Every agent, every API endpoint, every search query.

---

#### Table 2 — `chunks`

**What it stores:**
Every study gets split into overlapping pieces of text called chunks. Each chunk gets its own row here — with the text AND its vector embedding stored together.

**Example row:**

```
chunk_id    : uuid-123
nct_id      : NCT04788680       ← links back to studies table
chunk_text  : "SPONSOR: Novo Nordisk
               PRIMARY OUTCOME: Reduction in HbA1c at 26 weeks
               RESULTS POSTED: NO"
embedding   : [0.023, -0.041, 0.891, ...]  ← 1536 numbers
chunk_index : 0                 ← position in the document
source      : study             ← came from ClinicalTrials.gov
```

**Why we need it:**
This is the heart of semantic search. When an agent asks "find me studies where the sponsor never posted results", the system converts that question into 1536 numbers and finds the chunks whose numbers are most similar. This is called vector similarity search — and it only works because of this table.

**The special column:**
`embedding VECTOR(1536)` — this is what pgvector adds. A normal PostgreSQL column stores text or numbers. This column stores 1536 floating point numbers representing the meaning of the text. No other relational database can do this natively.

**Who writes to it:**
`vector_store.py` during processing — after chunker.py and embedder.py have done their work.

**Who reads from it:**
Every agent via the search tools. This is the most frequently queried table in the entire system.

---

#### Table 3 — `signals`

**What it stores:**
Every finding that an agent generates. When the Missing Results agent finds a completed trial with no results posted — that finding is a signal. It gets one row here.

**Example row:**

```
signal_id   : uuid-456
nct_id      : NCT04788680
agent       : missing_results_agent
signal_type : missing_results
summary     : "This trial completed in 2019 and has never posted
               results. It is now 5 years overdue."
evidence    : ["STATUS: COMPLETED", "RESULTS POSTED: NO",
               "Completion date: August 2019"]
confidence  : 0.92
status      : approved
created_at  : 2024-03-15 14:32:11
```

**Why we need it:**
This is the output of the entire system. Everything MOSAIC does — ingestion, processing, memory, agents — ultimately produces rows in this table. The FastAPI `/signals` endpoint reads from here. The human reviewer reads from here. This is where intelligence lives.

**Status values:**

* `pending` → signal generated, waiting
* `approved` → human confirmed it is real
* `rejected` → human said it was wrong
* `edited` → human corrected the summary

**Who writes to it:**
The analysis router in `api/routers/analysis.py` after agents finish running.

**Who reads from it:**
`GET /api/v1/signals` endpoint, the HITL review interface, the final intelligence brief.

---

#### Table 4 — `hitl_reviews`

**What it stores:**
Every human review decision. When an analyst looks at a signal and approves, rejects, or edits it — that decision gets recorded here permanently.

**Example row:**

```
review_id        : uuid-789
signal_id        : uuid-456       ← links back to signals table
reviewer         : chirantan@tarkaupskilling.com
decision         : rejected
rejection_reason : "This trial was terminated early due to COVID —
                   missing results for terminated trials is expected
                   and not a compliance violation"
reviewed_at      : 2024-03-15 15:00:00
```

**Why we need it:**
Two reasons. First, audit trail — we always know who reviewed what and when. Second, and most importantly — the rejection reason feeds back into the agent's procedural memory. The agent reads this and reasons differently in all future sessions. This is the learning loop. Without this table, the system cannot get smarter from human feedback.

**Who writes to it:**
`PATCH /api/v1/review/{queue_id}` endpoint when an analyst submits a decision.

**Who reads from it:**
`procedural_store.py` — reads rejection reasons to update agent reasoning rules.

---

#### Table 5 — `sponsor_profiles`

**What it stores:**
Accumulated knowledge about every research sponsor the system has ever encountered. Built up over time as agents analyse more studies.

**Example row:**

```
sponsor           : Novo Nordisk
credibility_score : 0.82
total_studies     : 47
results_posted    : 41
avg_delay_days    : 28.5
broken_promises   : 2
last_updated      : 2024-03-15 14:32:11
```

**Why we need it:**
This is the semantic memory layer for sponsors. When the Track Record agent analyses a new Novo Nordisk trial, it first checks this table — "what do we already know about this sponsor?" A sponsor with `credibility_score: 0.2` and `broken_promises: 8` is very different from one with `credibility_score: 0.95` and `broken_promises: 0`. Without this table, every agent run starts from zero knowledge. With it, the system accumulates intelligence over time.

**How credibility score is calculated:**

```
70% weight → results compliance rate (posted / total)
30% weight → promise keeping (penalised per broken promise)
Range: 0.0 (worst) to 1.0 (best)
```

**Who writes to it:**
`semantic_store.py` — updated after every agent analysis session.

**Who reads from it:**
Track Record agent, Pattern Finder agent, `GET /api/v1/sponsors/{name}` endpoint.

---

### Summary Table

| Table                | One Line                               | Written by          | Read by            |
| -------------------- | -------------------------------------- | ------------------- | ------------------ |
| `studies`          | Master record of every clinical trial  | Ingestion pipeline  | Every agent        |
| `chunks`           | Text + 3072-number embedding per chunk | Processing pipeline | Every search tool  |
| `signals`          | Every finding from every agent         | Analysis router     | API endpoints      |
| `hitl_reviews`     | Every human review decision            | Review router       | Procedural memory  |
| `sponsor_profiles` | Credibility profile per sponsor        | Semantic store      | Track record agent |

---

### The Flow in One Picture

```
ClinicalTrials.gov
        ↓
   studies table        ← ingestion writes here first
        ↓
   chunks table         ← processing writes embeddings here
        ↓
   agents run
        ↓
   signals table        ← agents write findings here
        ↓
   human reviews
        ↓
   hitl_reviews table   ← human decisions written here
        ↓
   sponsor_profiles     ← updated after every analysis
```

Every table serves a specific purpose. Remove any one of them and a layer of the system breaks. Together they form the complete data model for a production multi-agent intelligence system.
