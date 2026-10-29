# unit testing broken_promises_agent in isolation

# this agent is complex as
    # it calls multiple tools
    # it interacts with different memory stores (databases)

# mocking strategy
    # mock the LLM
    # mock the memory stores (EpisodicStore, ProceduralStore)
    # mock the tool functions

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock

from graph.state import GraphState, SignalOutput
from agents.broken_promises_agent import broken_promises_node

@pytest.fixture
def mock_llm():
    """
    provides a mocked LLM instance in broken_promises_agent to avoid actual API calls
    """

    with patch("agents.broken_promises_agent._llm", new_callable=AsyncMock) as mock:
        yield mock

@pytest.fixture
def mock_procedural_store():
    """
    mocks the ProceduralStore instance
    """

    with patch("agents.broken_promises_agent._procedural") as mock:
        # return some basic rules
        mock.get_procedures = AsyncMock(return_value=["Rule 1: Check primary outcomes", "Rule 2: Check timing"])
        yield mock

@pytest.fixture
def mock_episodic_store():
    """
    mocks the EpisodicStore instance
    """

    with patch("agents.broken_promises_agent._episodic") as mock:
        mock.save_episode = AsyncMock()
        yield mock

@pytest.fixture
def base_state():
    """
    provides a basic GraphState for testing
    """

    return {
        "task": "Detect outcome switching in NCT01234567",
        "nct_ids": ["NCT01234567"],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

@pytest.mark.asyncio
async def test_broken_promises_no_signals(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests the path where broken_promises_agent finds no signals (no violations)
    """

    # mock LLM to return NO_SIGNALS_FOUND
    mock_response = MagicMock()
    mock_response.content = "NO_SIGNALS_FOUND"
    mock_response.tool_calls = []
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await broken_promises_node(base_state)

    # verifying LLM was called
    mock_llm.ainvoke.assert_called_once()
    # verifying episodic memory was updated
    mock_episodic_store.save_episode.assert_called_once()
    # verifying state update
    assert "broken_promises_agent" in result["agents_activated"]
    assert len(result["signals"]) == 0

@pytest.mark.asyncio
async def test_broken_promises_detects_signal(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent correctly parses a signal from the LLM's output
    """

    signal_json = {
        "nct_id": "NCT01234567",
        "signal_type": "broken_promise",
        "summary": "Primary outcome changed from HbA1c to Weight loss after enrollment!",
        "evidence": ["Original: HbA1c", "New: Weight loss"],
        "confidence": 0.85
    }

    mock_response = MagicMock()
    mock_response.content = f"I found a violation: {json.dumps(signal_json)}"
    mock_response.tool_calls = []
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await broken_promises_node(base_state)

    assert len(result["signals"]) == 1
    assert result["signals"][0]["nct_id"] == "NCT01234567"
    assert result["signals"][0]["confidence"] == 0.85
    assert "broken_promises_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_broken_promises_tool_use_loop(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests broken_promises_agent's ability to call tools and process the results in a loop
    """

    # iteration 1: LLM calls get_study_amendments
    response1 = MagicMock()
    response1.content = "I need to check amendments."
    response1.tool_calls = [{
        "name": "get_study_amendments",
        "args": {"nct_id": "NCT01234567"}
    }]

    # iteration 2: LLM sees amendment data and generates a signal
    signal_json = {
        "nct_id": "NCT01234567",
        "signal_type": "broken_promise",
        "summary": "Outcome switched mid-study",
        "evidence": ["Amendment filed 6 months after start"],
        "confidence": 0.9
    }
    response2 = MagicMock()
    response2.content = f"Based on the data: {json.dumps(signal_json)}"
    response2.tool_calls = []

    mock_llm.ainvoke.side_effect = [response1, response2]

    # mocking the tool execution within the agent
    with patch("agents.broken_promises_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Amendment found: Primary outcome changed from X to Y."

        result = await broken_promises_node(base_state)

    # verifying tool was called
    mock_execute.assert_called_once()
    # verifying loop ran twice
    assert mock_llm.ainvoke.call_count == 2
    # verifying signal was captured
    assert len(result["signals"]) == 1
    assert result["signals"][0]["confidence"] == 0.9

@pytest.mark.asyncio
async def test_broken_promises_error_handling(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent gracefully handles exceptions and logs them in the state
    """

    # mock LLM to raise an exception
    mock_llm.ainvoke.side_effect = Exception("LLM Timeout Error!")

    result = await broken_promises_node(base_state)

    # verifying the error is captured in the error_log in the updated state
    assert any("broken_promises_agent" in err and "LLM Timeout Error!" in err for err in result["error_log"])
    # verifying that broken_promises_agent is still marked as activated
    assert "broken_promises_agent" in result["agents_activated"]