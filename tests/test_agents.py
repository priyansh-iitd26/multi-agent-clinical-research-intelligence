# testing framework
import pytest

# importing tools to create mocks
# allows to simulate database or API calls
# without actually making them
from unittest.mock import AsyncMock, MagicMock, patch

from agents.supervisor import supervisor_route, supervisor_compile
from graph.state import GraphState

# initializing the supervisor
@pytest.mark.asyncio
async def test_supervisor_route_initialization():
    """
    verifies that supervisor_route correctly initializes the state for agents
    """

    state = {
        "task": "Check for research integrity issues in NCT123",
        "nct_ids": ["NCT123"],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

    result = await supervisor_route(state)

    assert "run_id" in result
    assert result["agents_activated"] == []
    assert result["signals"] == []
    assert result["run_complete"] is False
    assert result["error_log"] == []

# compiling signals from agents
@pytest.mark.asyncio
async def test_supervisor_compile_aggregates_signals():
    """
    verifies that supervisor_compile correctly processes signals from multiple agents
    """

    state = {
        "task": "Check for research integrity issues in NCT123",
        "signals": [
            {"agent": "missing_results_agent", "signal_type": "missing_results", "nct_id": "NCT123", "confidence": 0.8, "summary": "Missing results found"},
            {"agent": "side_effect_agent", "signal_type": "side_effect", "nct_id": "NCT123", "confidence": 0.4, "summary": "Possible side effect discrepancy"},
        ],
        "agents_activated": ["missing_results_agent", "side_effect_agent"],
        "run_id": "test-run-123"
    }

    # mocking the LLM
    # intercepts the actual LLM with a fake AsyncMock object
    with patch('agents.supervisor.llm', new_callable=AsyncMock) as mock_llm:
        mock_response = MagicMock()
        
        mock_response.content = "Compiled Intelligence Brief: Found missing results and potential side effects."
        mock_llm.ainvoke.return_value = mock_response

        result = await supervisor_compile(state)

        assert result["run_complete"] is True
        assert result["total_signals"] == 2
        assert result["signals_requiring_review"] == 1
        assert "final_brief" in result
        assert "Compiled Intelligence Brief" in result["final_brief"]
        assert "missing_results_agent" in result["agents_activated"]
        assert "side_effect_agent" in result["agents_activated"]

# final test for edge case
@pytest.mark.asyncio
async def test_supervisor_compile_no_signals():
    """
    verifies supervisor_compile handles the case where no signals were found
    """
    
    state = {
        "task": "Check for research integrity issues in NCT123",
        "signals": [],
        "agents_activated": ["missing_results_agent", "side_effect_agent"],
        "run_id": "test-run-123"
    }

    result = await supervisor_compile(state)

    assert result["run_complete"] is True
    assert "EXECUTIVE SUMMARY" in result["final_brief"]
    assert "No significant research integrity signals were detected" in result["final_brief"]
