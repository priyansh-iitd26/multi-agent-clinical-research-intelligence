# defines the request and response shapes for every FastAPI endpoint
# endpoint : client -> request -> FastAPI endpoint -> response -> client

# what is a schema ?
# a schema is a contract - it just says:
# "when you call this endpoint, send data in this (request schema) shape.
# "when I respond, I will always respond in that (response schema) shape."

# without schemas, APIs are chaotic - callers never know what to send
# or what they will get back 
# with Pydantic schemas, everything is:
# - TYPED: every field has a declared type
# - VALIDATED: Pydantic rejects wrong types immediately with clear errors
# - DOCUMENTED: FastAPI reads these schemas and auto-generates Swagger docs
# - SERIALIZABLE: Pydantic converts to/from JSON automatically

# every API endpoint declares its request and response schema
# FastAPI automatically:
# - validates incoming request data against the request schema
# - serializes outgoing response data using the response schema
# - generates Swagger documentation at /docs showing all schemas

from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field

# request schemas

class AnalysisRequest(BaseModel):
    """
    request schema for POST /api/v1/analyze
    this is what is called to trigger a full analysis run
    all fields have defaults, so the simplest call is : POST /api/v1/analyze {}
    """

    task : str = Field(
        default="Find completed clinical trials with research integrity issues",
        description="The analysis task in plain English. Agents read this and decide what to investigate.",
        examples=["Find completed trials where sponsor never posted results"]
    )

    nct_ids : list[str] = Field(
        default=[],
        description="Specific NCT IDs to analyse. Empty list means analyse broadly across all studies.",
        examples=["NCT04788680", "NCT02208921"]
    )

    max_studies : int = Field(
        default=10,
        ge=1, 
        le=100,
        description="Maximum studies to analyse per agent. Default is 10."
    )

class ReviewDecisionRequest(BaseModel):
    """
    request schema for PATCH /api/v1/review/{queue_id}
    submitted by a human analyst after reviewing a queued signal
    """

    decision : str = Field(
        ...,
        description="The reviewer's decision: approve, reject, or edit.",
        examples=["reject", "approve", "edit"]
    )

    reviewer : str = Field(
        default="analyst",
        description="Name or ID of the human reviewer.",
        examples=["priyanshkumar.iitdelhi@gmail.com"]
    )

    rejection_reason : str = Field(
        default="",
        description="Why this signal was rejected. "
                    "IMPORTANT: This gets written to procedural memory "
                    "and permanently changes how the agent reasons. "
                    "Be specific and clear.",
        examples=["This trial was terminated early due to COVID — "
                "terminated trials are exempt from result posting requirements."]
    )

    edit_summary : str = Field(
        default="",
        description="Corrected signal summary if decision is 'edit'. "
                    "Replaces the agent's original summary."
    )

# response schemas

class SignalResponse(BaseModel):
    """
    one signal in an API response
    matches the structure of a row in signals table
    """

    signal_id: str
    nct_id: str
    agent: str
    signal_type: str
    summary: str
    confidence: float
    status: str
    created_at: datetime

    class Config:
        # Pydantic is allowed to construct response model
        # from an object's attribute, not only from dicts
        # (asyncpg returns Record objects)
        from_attributes = True

class AnalysisResponse(BaseModel):
    """
    response schema for POST /api/v1/analyze
    contains the complete results of one analysis run
    """

    run_id: str
    task: str
    final_brief: str
    total_signals: int
    signals_requiring_review: int
    agents_activated: list[str]
    duration_seconds: float

class ReviewQueueItem(BaseModel):
    """
    one item in the human review queue (hitl_reviews)
    response schema for GET /api/v1/review/queue
    """

    review_id: str
    signal_id: str
    agent: str
    signal_type: str
    summary: str
    confidence: float
    nct_id: str
    decision: str

class ReviewQueueResponse(BaseModel):
    """
    full response from GET /api/v1/review/queue
    contains the queue + summary statistics
    """

    queue: list[ReviewQueueItem]
    total_pending: int
    total_approved: int
    total_rejected: int

class ReviewDecisionResponse(BaseModel):
    """
    response schema for PATCH /api/v1/review/{queue_id}
    confirms if the decision was recorded
    """

    success: bool
    decision: str
    signal_id: str
    queue_id: str
    memory_updated: bool
    # True if this was a rejection - meaning procedural memory
    # was updated with the rejection reason (the learning loop)
    message: str

class EpisodeResponse(BaseModel):
    """
    one episode from episodic memory
    response schema for GET /api/v1/memory/episodes
    """

    episode_id: str
    agent_name: str
    nct_id: str | None = None
    content: str
    outcome: str | None = None
    similarity: float | None = None
    # similarity is only present when the episode was found
    # via semantic search - not when listing recent episodes
    created_at:  datetime

class ProcedureResponse(BaseModel):
    """
    one reasoning rule from procedural memory
    response schema for GET /api/v1/memory/procedures/{agent_name}
    """

    procedure_id: str
    agent_name: str
    rule_text: str
    rule_type: str
    source: str
    created_at: datetime

class SponsorProfileResponse(BaseModel):
    """
    full sponsor credibility profile
    response schema for GET /api/v1/sponsors/{sponsor_name}
    """

    sponsor: str
    credibility_score: float
    total_studies: int
    results_posted: int
    results_missing: int
    broken_promises: int
    avg_delay_days: float
    last_updated: datetime

class HealthResponse(BaseModel):
    """
    system health check response
    response schema for GET /api/v1/health
    """

    status: str
    # "healthy" if all systems operational, "degraded" if issues
    app: str
    version: str
    database: str
    # "connected" or "disconnected"
    details: dict[str, Any]
    # contains: signals_in_db, pending_reviews, episodes_count etc.