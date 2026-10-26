# validating two things
    # whether the supervisor correctly updates the state to trigger the six specialists
    # whether the supervisor passes the gathered signals to the LLM for brief compilation

# mocking of client ONLY
# no tools, no database(s) calls here!

# coverage:
    # supervisor_route
    # supervisor_compile (no signals)
    # supervisor_compile (with signals)
    # supervisor_compile (fallback)

import pytest
import uuid
from unittest.mock import patch, MagicMock, AsyncMock
from graph.state import GraphState, SignalOutput
from agents.supervisor import supervisor_route, supervisor_compile

@pytest.fixture
def mock_llm():
    """
    provides a mocked LLM instance to avoid actual API calls
    """

    with patch("agents.supervisor.llm") as mock:
        # setup the mock to return a MagicMock
        mock_response = MagicMock()
        mock_response.content = "Mocked Final Intelligence Brief!"
        mock.ainvoke = AsyncMock(return_value=mock_response)
        yield mock

@pytest.fixture
def base_state():
    """
    provides a basic GraphState for testing
    """

    return {
        "task": "Analyze NCT01234567 for missing results",
        "nct_ids": ["NCT01234567"],
        "max_studies": 1,
        "messages": [],
        "signals": [],
        "agents_activated": [],
        "error_log": [],
        "final_brief": "",
        "run_complete": False,
        "run_id": ""
    }

@pytest.mark.asyncio
async def test_supervisor_route(base_state):
    """
    tests that supervisor_route correctly initializes the run:
    - generates a run_id
    - sets run_complete to False
    - initializes empty lists for signals and agents_activated
    """

    result = await supervisor_route(base_state)

    # verifying whether run_id is a valid uuid string
    assert "run_id" in result
    assert uuid.UUID(result["run_id"]) # should not raise ValueError
    
    assert result["run_complete"] is False
    assert result["agents_activated"] == []
    assert result["signals"] == []
    assert result["error_log"] == []

@pytest.mark.asyncio
async def test_supervisor_compile_no_signals(base_state):
    """
    tests that supervisor_compile returns a clean summary when no signals are found
    """

    # ensuring that signals is empty list
    base_state["signals"] == []
    base_state["agents_activated"] = ["missing_results_agent"]

    result = await supervisor_compile(base_state)

    assert "EXECUTIVE SUMMARY" in result["final_brief"]
    assert "No significant research integrity signals were detected" in result["final_brief"]
    assert result["run_complete"] is True
    assert result["agents_activated"] == ["missing_results_agent"]

@pytest.mark.asyncio
async def test_supervisor_compile_signals(base_state, mock_llm):
    """
    tests that supervisor_compile correctly invokes the LLM when signals are present
    """

    # sample signals
    signals = [
        {
            "agent": "missing_results_agent",
            "signal_type": "missing_results",
            "nct_id": "NCT01234567",
            "summary": "Trial ended 2 years ago but no results posted",
            "evidence": ["Link to ClinicalTrials.gov"],
            "confidence": 0.8
        },
        {
            "agent": "broken_promises_agent",
            "signal_type": "broken_promise",
            "nct_id": "NCT01234567",
            "summary": "Primary endpoint changed mid-study",
            "evidence": ["Change log"],
            "confidence": 0.4
        }
    ]

    base_state["signals"] = signals
    base_state["agents_activated"] = ["missing_results_agent", "broken_promises_agent"]
    base_state["task"] = "Check NCT01234567"

    result = await supervisor_compile(base_state)

    # verifying if the LLM was called
    mock_llm.ainvoke.assert_called_once()

    # verifying state updates
    assert result["final_brief"] == "Mocked Final Intelligence Brief!"
    assert result["run_complete"] is True
    assert result["total_signals"] == 2
    assert result["signals_requiring_review"] == 1 # only the 0.4 confidence one
    assert result["agents_activated"] == ["missing_results_agent", "broken_promises_agent"]

@pytest.mark.asyncio
async def test_supervisor_compile_llm_failure_fallback(base_state, mock_llm):
    """
    tests that supervisor_compile falls back to a structured plain-text brief if the LLM call fails
    """

    # sample signals
    signals = [{
        "agent": "pattern_finder_agent",
        "signal_type": "cross_study_pattern",
        "nct_id": "NCT999",
        "summary": "Consistent delay across sponsor trials",
        "confidence": 0.7,
        "evidence": []
    }]

    base_state["signals"] = signals
    base_state["agents_activated"] = ["pattern_finder_agent"]

    # mock llm to raise an exception
    mock_llm.ainvoke.side_effect = Exception("API Connection Error!")

    result = await supervisor_compile(base_state)

    # verifying if the fallback brief is used
    assert "EXECUTIVE SUMMARY" in result["final_brief"]
    assert "NCT999" in result["final_brief"]
    assert "generated without LLM assistance" in result["final_brief"]
    assert result["run_complete"] is True