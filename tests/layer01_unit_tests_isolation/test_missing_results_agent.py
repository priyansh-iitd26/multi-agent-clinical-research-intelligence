# unit testing missing_results_agent in isolation

# this agent identifies trials that are completed but failed to post results

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from graph.state import GraphState, SignalOutput
from agents.missing_results_agent import missing_results_node

@pytest.fixture
def mock_llm():
    """
    mocks the LLM instance for missing_results_agent
    """

    with patch("agents.missing_results_agent._llm", new_callable=AsyncMock) as mock:
        yield mock

@pytest.fixture
def mock_procedural_store():
    """
    mocking ProceduralStore for reasoning rule retrieval
    """

    with patch("agents.missing_results_agent._procedural") as mock:
        mock.get_procedures = AsyncMock(return_value=[
            "Flag ONLY IF status is COMPLETED and results_posted is FALSE",
            "Exclude terminated trials"
        ])
        yield mock

@pytest.fixture
def mock_episodic_store():
    """
    mocks the EpisodicStore instance
    """

    with patch("agents.missing_results_agent._episodic") as mock:
        mock.save_episode = AsyncMock()
        yield mock

@pytest.fixture
def base_state():
    """
    basic GraphState for testing
    """

    return {
        "task": "Investigate missing results for NCT00000000",
        "nct_ids": ["NCT00000000"],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

@pytest.mark.asyncio
async def test_missing_results_no_signals(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    testing the path where no signals/violations were found by missing_results_agent
    """

    mock_response = MagicMock()
    mock_response.content = "NO_SIGNALS_FOUND"
    mock_response.tool_calls = []
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await missing_results_node(base_state)

    mock_llm.ainvoke.assert_called_once()
    mock_episodic_store.save_episode.assert_called_once()
    assert "missing_results_agent" in result["agents_activated"]
    assert len(result["signals"]) == 0

@pytest.mark.asyncio
async def test_missing_results_detects_signal(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that missing_results_agent correctly parses a signal from the LLM's output 
    """
    signal_json = {
        "nct_id": "NCT00000000",
        "signal_type": "missing_results",
        "summary": "Completed 3 years ago, zero results posted.",
        "evidence": ["Completion date: 2021-01", "results_posted: False"],
        "confidence": 0.9
    }

    mock_response = MagicMock()
    mock_response.content = f"Violation found: {json.dumps(signal_json)}"
    mock_response.tool_calls = []
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await missing_results_node(base_state)

    assert len(result["signals"]) == 1
    assert result["signals"][0]["nct_id"] == "NCT00000000"
    assert result["signals"][0]["confidence"] == 0.9

@pytest.mark.asyncio
async def test_missing_results_comprehensive_tool_loop(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests the full agentic cycle:
        - LLM decides to check results status.
        - LLM decides to check sponsor profile.
        - LLM decides to update sponsor profile based on fraud.
        - LLM generates final signal.
    """
    
    # iteration 1: check results
    res1 = MagicMock(content="Checking results...", tool_calls=[{"name": "check_results_posted", "args": {"nct_id":
"NCT00000000"}}])
    
    # iteration 2: check sponsor
    res2 = MagicMock(content="Checking sponsor...", tool_calls=[{"name": "get_sponsor_profile", "args":
{"sponsor_name": "Bad Pharma"}}])
    
    # iteration 3: update sponsor
    res3 = MagicMock(content="Updating sponsor profile...", tool_calls=[{"name": "update_sponsor_profile", "args":
{"sponsor_name": "Bad Pharma", "results_posted": False}}])
    
    # iteration 4: final signal generated
    sig = {"nct_id": "NCT00000000", "signal_type": "missing_results", "summary": "Missing results", "evidence": [],
"confidence": 0.8}
    res4 = MagicMock(content=f"Final: {json.dumps(sig)}", tool_calls=[])

    # successive invocations return different responses
    mock_llm.ainvoke.side_effect = [res1, res2, res3, res4]

    # tool mocking within the agent
    with patch("agents.missing_results_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.side_effect = [
            '{"results_posted": false, "status": "COMPLETED"}',
            '{"credibility": 0.4}',
            '{"success": true}'
        ]

        result = await missing_results_node(base_state)

    assert mock_execute.call_count == 3
    assert mock_llm.ainvoke.call_count == 4
    assert len(result["signals"]) == 1
    assert result["signals"][0]["confidence"] == 0.8

@pytest.mark.asyncio
async def test_missing_results_malformed_json(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests resilience against bad JSON syntax
    """

    mock_response = MagicMock()
    mock_response.content = "Found something: { nct_id: NCT123, confidence: 0.7 [MISSING BRACKET]"
    mock_response.tool_calls = []
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await missing_results_node(base_state)
    assert len(result["signals"]) == 0
    assert "missing_results_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_missing_results_tool_failure_recovery(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that tool errors don't crash missing_results_agent
    """

    res1 = MagicMock(content="Checking...", tool_calls=[{"name": "check_results_posted", "args": {"nct_id": "NCT1" }}])
    res2 = MagicMock(content="NO_SIGNALS_FOUND", tool_calls=[])

    mock_llm.ainvoke.side_effect = [res1, res2]

    with patch("agents.missing_results_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Error: API Connection Timeout"
        result = await missing_results_node(base_state)

    assert len(result["signals"]) == 0
    assert "missing_results_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_missing_results_all_tools_invoked(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    stress testing: verifies that the missing_result_agent can successfully request and
    execute every single tool in its AGENT_TOOLS registry
    """

    # defining the expected sequence of tool calls
    tool_sequence = [
        {"name": "search_past_episodes", "args": {"query": "missing results", "agent_name": "missing_results_agent"}},
        {"name": "search_studies_by_meaning", "args": {"query": "completed trials missing results"}},
        {"name": "fetch_study_details", "args": {"nct_id": "NCT123"}},
        {"name": "check_results_posted", "args": {"nct_id": "NCT123"}},
        {"name": "get_sponsor_profile", "args": {"sponsor_name": "Bad Pharma"}},
        {"name": "update_sponsor_profile", "args": {"sponsor_name": "Bad Pharma", "results_posted": False}},
        {"name": "save_episode", "args": {"agent_name": "missing_results_agent", "content": "Found violation"}}
    ]

    # mock LLM to request tools one by one
    responses = []

    for tool in tool_sequence:
        responses.append(MagicMock(content=f"Calling {tool['name']}...", tool_calls=[tool]))

    # final response to stop the loop
    responses.append(MagicMock(content="NO_SIGNALS_FOUND", tool_calls=[]))

    mock_llm.ainvoke.side_effect = responses

    with patch("agents.missing_results_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = '{"status": "success"}'

        result = await missing_results_node(base_state)

    # verifying that every tool in the sequence was called
    assert mock_execute.call_count == len(tool_sequence)

    # verifying the specific order and names of tool calls
    called_tool_names = [call[0][0]["name"] for call in mock_execute.call_args_list]

    assert called_tool_names == [
        tool["name"] for tool in tool_sequence
    ]