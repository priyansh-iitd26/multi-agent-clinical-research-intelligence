# handles POST /api/v1/analyze endpoint that triggers a full system run

# 1. recieves the analysis task and parameters
# 2. invokes the compiled LangGraph `graph` with the task
# 3. the `graph` runs all 6 specialist agents in parallel
# 4. signals are processed through HITL
# 5. supervisor compiles the final brief
# 6. returns the complete analysis to the caller

import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import AnalysisRequest, AnalysisResponse
from api.dependencies import get_hitl, get_graph
from graph.hitl import HITL

from config.logging_config import setup_logging

logger = setup_logging(__name__)

router = APIRouter()
# APIRouter() => a modular group of related API endpoints

@router.post(
    "/analyze", 
    response_model=AnalysisResponse,
    summary="Trigger a full analysis run of the system",
    description="Runs all 6 specialist agents in parallel and returns "
                "a compiled intelligence brief with all signals found.",
    tags=["Analysis"]
)
async def run_analysis(
    request: AnalysisRequest,
    hitl: HITL = Depends(get_hitl),
    graph = Depends(get_graph)
):
    """
    triggers a complete analysis run of the system

    1. recieves the analysis task and parameters
    2. invokes the compiled LangGraph `graph` with the task
    3. the `graph` runs all 6 specialist agents in parallel
    4. signals are processed through HITL
    5. supervisor compiles the final brief
    6. returns the complete analysis to the caller
    """

    run_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(
        f"Analysis run started! | "
        f"run_id = {run_id} | "
        f"task = '{request.task[:80]}'"
    )

    try:
        # step - 1
        # building the initial state dict
        # which get passed to the first node in `graph`
        # every field in GraphState must have an initial value here
        # LangGraph agents read from and write to this state as they run
        initial_state = {
            "task": request.task,
            "nct_ids": request.nct_ids,
            "max_studies": request.max_studies,
            "messages": [],
            "signals": [],
            "agents_activated": [],
            "final_brief": "",
            "run_complete": False,
            "run_id": run_id,
            "error_log": []
        }

        # step - 2 & 3
        # ainvoke() is the async version of invoke()
        # this runs the entire graph -> all agents - and waits for completion
        # the graph runs supervisor_route → all 6 specialists in parallel
        # → supervisor_compile → returns final state
        result = await graph.ainvoke(initial_state)

        # step - 4, 5 & 6
        # processing signals through HITL
        signals = result.get("signals", [])
        processed_signals = []

        for signal in signals:
            # routing each signal through HITL
            # high confidence -> saved directly to signals table
            # low confidence -> sent to review queue (hitl_reviews)
            hitl_result = await hitl.process_signal(signal)

            processed_signals.append({
                **signal,
                "hitl_action": hitl_result.get("action")
            })

            logger.info(
                f"Signal saved directly | "
                f"agent = {signal.get('agent')} | "
                f"confidence = {signal.get('confidence', 0):.2f}"
            )

        duration = round(time.time() - start_time, 2)

        num_signals_requiring_review = sum(
            1 for s in processed_signals
            if s.get("hitl_action") == "sent_to_review"
        )

        logger.info(
            f"Analysis run completed! | "
            f"run_id = {run_id} | "
            f"signals = {len(signals)} | "
            f"duration = {duration}s"
        )

        return AnalysisResponse(
            run_id = run_id,
            task = request.task,
            final_brief = result.get("final_brief", "No brief generated."),
            total_signals = len(signals),
            signals_requiring_review = num_signals_requiring_review,
            agents_activated = result.get("agents_activated", []),
            duration_seconds = duration
        )

    except Exception as e:
        logger.error(f"Analysis run failed! | run_id = {run_id} | error = {e}")
        raise HTTPException(
            status_code = 500,
            detail = f"Analysis run failed: {str(e)}"
            # HTTPException tells FastAPI to return an error response
            # status_code = 500 => Internal Server Error
        )