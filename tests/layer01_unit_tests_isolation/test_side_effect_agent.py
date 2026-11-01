# unit testing side_effect_agent in isolation

# this agent hunts for discrepancies between official sponsor filings
# and `independent` peer-reviewed research papers

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock

from graph.state import GraphState, SignalOutput
from agents.side_effect_agent import side_effect_node

@pytest.fixture
def mock_llm():
    with patch("agents.side_effect_agent._llm") as mock:
        mock.ainvoke = AsyncMock()
        yield mock

@pytest.fixture
def mock_procedural_store():
    with patch("agents.side_effect_agent._procedural") as mock:
        mock.get_procedures = AsyncMock(return_value=[
            "Prioritize patient safety - when in doubt, flag it",
            "Look for hospitalizations mentioned in papers but not in filings"
        ])
        yield mock

@pytest.fixture
def mock_episodic_store():
    with patch("agents.side_effect_agent._episodic") as mock:
        mock.save_episode = AsyncMock()
        yield mock

@pytest.fixture
def base_state():
    return {
        "task": "Find safety gaps for NCT00000000",
        "nct_ids": ["NCT00000000"],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

@pytest.mark.asyncio
async def test_side_effect_no_signals(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests the path where official filings and papers are in agreement
    """
    
    mock_response = MagicMock()
    mock_response.content = "NO_SIGNALS_FOUND"
    mock_response.tool_calls = []
    mock_llm.ainvoke.return_value = mock_response

    result = await side_effect_node(base_state)

    mock_llm.ainvoke.assert_called_once()
    mock_episodic_store.save_episode.assert_called_once()
    assert "side_effect_agent" in result["agents_activated"]
    assert len(result["signals"]) == 0

@pytest.mark.asyncio
async def test_side_effect_detects_safety_gap(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests parsing of a high-confidence safety gap signal by side_effect_agent
    """
    
    signal_json = {
        "nct_id": "NCT00000000",
        "signal_type": "safety_gap",
        "summary": "Filing claims no SAEs, but paper reports 2 cardiac arrests",
        "evidence": ["Filing: 'No serious adverse events'", "Paper: 'Two patients suffered MI'"],
        "confidence": 0.95
    }

    mock_response = MagicMock()
    mock_response.content = f"Safety gap detected: {json.dumps(signal_json)}"
    mock_response.tool_calls = []
    mock_llm.ainvoke.return_value = mock_response

    result = await side_effect_node(base_state)

    assert len(result["signals"]) == 1
    assert result["signals"][0]["confidence"] == 0.95
    assert result["signals"][0]["signal_type"] == "safety_gap"

@pytest.mark.asyncio
async def test_side_effect_comprehensive_tool_loop(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests the critical safety-check reasoning chain:
    check episodes -> fetch study -> compare filing vs papers -> generate signal
    """

    # sequence of LLM tool requests
    tool_calls = [
        {"name": "search_past_episodes", "args": {"query": "safety gaps", "agent_name": "side_effect_agent"}},
        {"name": "fetch_study_details", "args": {"nct_id": "NCT00000000"}},
        {"name": "compare_filing_vs_papers", "args": {"nct_id": "NCT00000000"}},
    ]

    responses = []
    for call in tool_calls:
        responses.append(MagicMock(content=f"Requesting {call['name']}...", tool_calls=[call]))

    # final signal
    sig = {"nct_id": "NCT00000000", "signal_type": "safety_gap", "summary": "Discrepancy found", "evidence": [], "confidence": 0.8}
    responses.append(MagicMock(content=f"Analysis complete: {json.dumps(sig)}", tool_calls=[]))

    mock_llm.ainvoke.side_effect = responses

    with patch("agents.side_effect_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        # simulating the 'compare_filing_vs_papers' tool returning a discrepancy
        mock_execute.side_effect = [
            '{"episodes": []}',
            '{"study_details": "..."}',
            '{"gap_detected": true, "filing": "No SAEs", "papers": "3 Hospitalizations"}'
        ]

        result = await side_effect_node(base_state)

    assert mock_execute.call_count == 3
    assert mock_llm.ainvoke.call_count == 4
    assert len(result["signals"]) == 1
    assert result["signals"][0]["confidence"] == 0.8

@pytest.mark.asyncio
async def test_side_effect_malformed_json(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests resilience against bad JSON in safety reports
    """
    
    mock_response = MagicMock()
    mock_response.content = "Violation: { nct_id: 'NCT123', confidence: 0.7 [MALFORMED]"
    mock_response.tool_calls = []
    mock_llm.ainvoke.return_value = mock_response

    result = await side_effect_node(base_state)
    
    assert len(result["signals"]) == 0
    assert "side_effect_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_side_effect_tool_failure(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that tool failures (e.g. PubMed down) are handled gracefully
    """
    
    res1 = MagicMock(content="Checking papers...", tool_calls=[{"name": "compare_filing_vs_papers", "args": {"nct_id": "NCT1"}}])
    res2 = MagicMock(content="NO_SIGNALS_FOUND", tool_calls=[])

    mock_llm.ainvoke.side_effect = [res1, res2]

    with patch("agents.side_effect_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.return_value = "Error: PubMed API Connection Failed!"
        
        result = await side_effect_node(base_state)

    assert len(result["signals"]) == 0
    assert "side_effect_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_side_effect_critical_exception(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that top-level agent crashes are caught and logged in state
    """
    
    mock_llm.ainvoke.side_effect = Exception("Asyncio Loop Error")
    result = await side_effect_node(base_state)

    assert any("side_effect_agent" in err and "Asyncio Loop Error" in err for err in result["error_log"])
    assert "side_effect_agent" in result["agents_activated"]
    