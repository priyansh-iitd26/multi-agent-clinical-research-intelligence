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

@pytest.mark.asyncio
async def test_broken_promises_malformed_json(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent handles malformed JSON from LLM without crashing
    """

    # mock LLM to return malformed JSON
    mock_response = MagicMock()
    mock_response.content = "I found a violation: { bad_json: 'missing quotes }"
    mock_response.tool_calls = []
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await broken_promises_node(base_state)

    # verifying no signals were added due to parsing error
    assert len(result["signals"]) == 0
    # verifying agent still marked as activated
    assert "broken_promises_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_broken_promises_tool_execution_failure(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent handles tool execution errors gracefully
    """

    # iteration 1: LLM calls get_study_amendments
    response1 = MagicMock()
    response1.content = "I need to check amendments."
    response1.tool_calls = [{
        "name": "get_study_amendments",
        "args": {"nct_id": "NCT01234567"}
    }]

    # iteration 2: LLM sees the error and decides it's inconclusive
    response2 = MagicMock()
    response2.content = "NO_SIGNALS_FOUND"
    response2.tool_calls = []

    mock_llm.ainvoke.side_effect = [response1, response2]

    # mocking the tool execution to fail
    with patch("agents.broken_promises_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Error: API rate limit exceeded!"

        result = await broken_promises_node(base_state)

    # verifying tool was called and failed
    mock_execute.assert_called_once()
    # verifying loop continued and eventually finished
    assert mock_llm.ainvoke.call_count == 2
    # verifying no signals were generated
    assert len(result["signals"]) == 0

@pytest.mark.asyncio
async def test_broken_promises_updates_sponsor_profile(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent calls update_sponsor_profile when fraud is detected
    """

    # iteration 1: LLM detects fraud and decides to update sponsor profile FIRST
    response1 = MagicMock()
    response1.content = "I detected outcome switching fraud. I need to update the sponsor profile."
    response1.tool_calls = [{
        "name": "update_sponsor_profile",
        "args": {"nct_id": "NCT01234567", "note": "Outcome switching detected"}
    }]

    # iteration 2: LLM receives tool success and outputs the final JSON signal
    signal_json = {
        "nct_id": "NCT01234567",
        "signal_type": "broken_promise",
        "summary": "Fraudulent outcome switch",
        "evidence": ["Evidence A"],
        "confidence": 0.95
    }
    
    response2 = MagicMock()
    # provide the JSON in the final iteration so the agent's parser catches it
    response2.content = f"Analysis complete. Here is the official signal: {json.dumps(signal_json)}"
    response2.tool_calls = []

    mock_llm.ainvoke.side_effect = [response1, response2]

    with patch("agents.broken_promises_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Sponsor profile updated successfully."

        result = await broken_promises_node(base_state)

    # verifying update_sponsor_profile was actually called
    called_tool_names = [call.args[0]["name"] for call in mock_execute.call_args_list]
    assert "update_sponsor_profile" in called_tool_names
    assert len(result["signals"]) == 1

@pytest.mark.asyncio
async def test_broken_promises_tool_search_studies_by_meaning(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent can use search_studies_by_meaning to find candidate trials
    """
    response1 = MagicMock()
    response1.content = "Searching for trials with outcome switching."
    response1.tool_calls = [{
        "name": "search_studies_by_meaning",
        "args": {"query": "primary outcome changed after enrollment"}
    }]

    response2 = MagicMock()
    response2.content = "NO_SIGNALS_FOUND"
    response2.tool_calls = []

    mock_llm.ainvoke.side_effect = [response1, response2]

    with patch("agents.broken_promises_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Found NCT09999999 with protocol amendments."
        result = await broken_promises_node(base_state)

    called_tool_names = [call.args[0]["name"] for call in mock_execute.call_args_list]
    assert "search_studies_by_meaning" in called_tool_names
    assert "broken_promises_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_broken_promises_tool_search_past_episodes(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent uses search_past_episodes to leverage previous analysis
    """
    response1 = MagicMock()
    response1.content = "Checking for similar past cases."
    response1.tool_calls = [{
        "name": "search_past_episodes",
        "args": {"query": "outcome switching", "agent_name": "broken_promises_agent"}
    }]

    response2 = MagicMock()
    response2.content = "NO_SIGNALS_FOUND"
    response2.tool_calls = []

    mock_llm.ainvoke.side_effect = [response1, response2]

    with patch("agents.broken_promises_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Past episode found: NCT08888888 had similar pattern."
        result = await broken_promises_node(base_state)

    called_tool_names = [call.args[0]["name"] for call in mock_execute.call_args_list]
    assert "search_past_episodes" in called_tool_names
    assert "broken_promises_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_broken_promises_tool_save_episode(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent uses save_episode tool via LLM if it decides to
    """
    response1 = MagicMock()
    response1.content = "Saving this interesting finding to memory."
    response1.tool_calls = [{
        "name": "save_episode",
        "args": {"agent_name": "broken_promises_agent", "episodic_content": "Found weird pattern in NCT0123", "outcome": "manual_save"}
    }]

    response2 = MagicMock()
    response2.content = "NO_SIGNALS_FOUND"
    response2.tool_calls = []

    mock_llm.ainvoke.side_effect = [response1, response2]

    with patch("agents.broken_promises_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Episode saved."
        result = await broken_promises_node(base_state)

    called_tool_names = [call.args[0]["name"] for call in mock_execute.call_args_list]
    assert "save_episode" in called_tool_names

@pytest.mark.asyncio
async def test_broken_promises_tool_get_sponsor_profile(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent checks sponsor credibility via get_sponsor_profile
    """
    response1 = MagicMock()
    response1.content = "Checking sponsor track record."
    response1.tool_calls = [{
        "name": "get_sponsor_profile",
        "args": {"nct_id": "NCT01234567"}
    }]

    response2 = MagicMock()
    response2.content = "NO_SIGNALS_FOUND"
    response2.tool_calls = []

    mock_llm.ainvoke.side_effect = [response1, response2]

    with patch("agents.broken_promises_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Sponsor has 2 previous warnings for integrity issues."
        result = await broken_promises_node(base_state)

    called_tool_names = [call.args[0]["name"] for call in mock_execute.call_args_list]
    assert "get_sponsor_profile" in called_tool_names

@pytest.mark.asyncio
async def test_broken_promises_tool_fetch_study_details(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that broken_promises_agent fetches basic study details
    """
    response1 = MagicMock()
    response1.content = "Fetching study details."
    response1.tool_calls = [{
        "name": "fetch_study_details",
        "args": {"nct_id": "NCT01234567"}
    }]

    response2 = MagicMock()
    response2.content = "NO_SIGNALS_FOUND"
    response2.tool_calls = []

    mock_llm.ainvoke.side_effect = [response1, response2]

    with patch("agents.broken_promises_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Study Details: Primary outcome is HbA1c."
        result = await broken_promises_node(base_state)

    called_tool_names = [call.args[0]["name"] for call in mock_execute.call_args_list]
    assert "fetch_study_details" in called_tool_names