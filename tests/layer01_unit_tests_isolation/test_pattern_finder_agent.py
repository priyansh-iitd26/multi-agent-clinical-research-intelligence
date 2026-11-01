# unit testing pattern_finder_agent in isolation

# this agent finds cross-study patterns across multiple trials and sponsors
# integrates data from database search, clinical APIs, and PubMed research

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock

from graph.state import GraphState, SignalOutput
from agents.pattern_finder_agent import pattern_finder_node

@pytest.fixture
def mock_llm():
    with patch("agents.pattern_finder_agent._llm") as mock:
        mock.ainvoke = AsyncMock()
        yield mock

@pytest.fixture
def mock_procedural_store():
    with patch("agents.pattern_finder_agent._procedural") as mock:
        mock.get_procedures = AsyncMock(return_value=[
            "A pattern requires atleast 3 studies to be meaningful",
            "Connect the dots across different sponsors for the same drug"
        ])
        yield mock

@pytest.fixture
def mock_episodic_store():
    with patch("agents.pattern_finder_agent._episodic") as mock:
        mock.save_episode = AsyncMock()
        yield mock

@pytest.fixture
def base_state():
    return {
        "task": "Find cross-study patterns for Semaglutide",
        "nct_ids": [],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

@pytest.mark.asyncio
async def test_pattern_finder_no_signals(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests the path where pattern_finder_agent finds no meaningful patterns
    """

    mock_response = MagicMock()
    mock_response.content = "NO_SIGNALS_FOUND"
    mock_response.tool_calls = []
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await pattern_finder_node(base_state)

    # verification
    mock_llm.ainvoke.assert_called_once()
    mock_episodic_store.save_episode.assert_called_once()
    assert "pattern_finder_agent" in result["agents_activated"]
    assert len(result["signals"]) == 0

@pytest.mark.asyncio
async def test_pattern_finder_detects_cross_study_signal(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that pattern_finder_agent correctly parses a cross-study pattern signal
    """

    # sample signal
    signal_json = {
        "nct_id": "PATTERN_001",
        "signal_type": "cross_study_pattern",
        "summary": "Three different sponsors all switched outcomes for Drug X",
        "evidence": ["Study A switched", "Study B switched", "Study C switched"],
        "confidence": 0.70
    }

    mock_response = MagicMock()
    mock_response.content = f"I have identified a systemic pattern: {json.dumps(signal_json)}"
    mock_response.tool_calls = []
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await pattern_finder_node(base_state)

    # verification
    assert len(result["signals"]) == 1
    assert result["signals"][0]["signal_type"] == "cross_study_pattern"
    assert result["signals"][0]["confidence"] == 0.70

@pytest.mark.asyncio
async def test_pattern_finder_all_tools_integration(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests pattern_finder_agent's ability to use its full toolset in a single reasoning chain
    search meaning -> PubMed search -> clinical details -> sponsor profile -> signal 
    """

    # mock sequence of LLM tool requests
    tool_calls = [
        {"name": "search_studies_by_meaning", "args": {"query": "Semaglutide patterns"}},
        {"name": "search_pubmed_by_query", "args": {"query": "Semaglutide cross-trial analysis"}},
        {"name": "fetch_study_details", "args": {"nct_id": "NCT123"}},
        {"name": "get_sponsor_profile", "args": {"sponsor_name": "Global Pharma"}}
    ]

    responses = []
    for call in tool_calls:
        responses.append(MagicMock(content=f"Requesting {call['name']}...", tool_calls=[call]))

    # final result
    sig = {"nct_id": "PATTERN_X", "signal_type": "cross_study_pattern", "summary": "systemic fraud", "evidence": [], "confidence": 0.9}
    responses.append(MagicMock(content=f"Final Analysis: {json.dumps(sig)}", tool_calls=[]))

    mock_llm.ainvoke.side_effect = responses

    with patch("agents.pattern_finder_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = '{"status": "sucess", "data": "some evidence"}'

        result = await pattern_finder_node(base_state)

    # verifying every tool call in the sequence
    assert mock_execute.call_count == len(tool_calls)
    
    # verifying the specific order and names of tool calls
    called_tool_names = [call[0][0]["name"] for call in mock_execute.call_args_list]
    assert called_tool_names == [
        tool["name"] for tool in tool_calls
    ]

    assert len(result["signals"]) == 1
    assert result["signals"][0]["confidence"] == 0.9

@pytest.mark.asyncio
async def test_pattern_finder_malformed_json(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests resilience against bad JSON syntax in pattern reports
    """
    
    mock_response = MagicMock()
    mock_response.content = "Found pattern: { nct_id: 'P1', confidence: 0.8 [BAD JSON]"
    mock_response.tool_calls = []
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    result = await pattern_finder_node(base_state)

    assert len(result["signals"]) == 0
    assert "pattern_finder_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_pattern_finder_tool_failure(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that pattern_finder_agent doesn't crash when a tool (like PubMed) returns an error
    """
    
    res1 = MagicMock(content="Searching PubMed...", tool_calls=[{"name": "search_pubmed_by_query", "args":
{"query": "test"}}])
    res2 = MagicMock(content="NO_SIGNALS_FOUND", tool_calls=[])
    mock_llm.ainvoke.side_effect = [res1, res2]

    with patch("agents.pattern_finder_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Error: PubMed API Timeout"

        result = await pattern_finder_node(base_state)

    assert len(result["signals"]) == 0
    assert "pattern_finder_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_pattern_finder_critical_error(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that top-level exceptions are captured in the error_log
    """
    
    mock_llm.ainvoke.side_effect = Exception("GPU Out of Memory")
    result = await pattern_finder_node(base_state)
    
    assert any("pattern_finder_agent" in err and "GPU Out of Memory" in err for err in result["error_log"])