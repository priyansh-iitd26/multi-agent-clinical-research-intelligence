# unit testing timeline_agent in isolation

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock

from graph.state import GraphState, SignalOutput
from agents.timeline_agent import timeline_node

@pytest.fixture
def mock_llm():
    with patch("agents.timeline_agent._llm") as mock:
        mock.ainvoke = AsyncMock()
        yield mock

@pytest.fixture
def mock_procedural_store():
    with patch("agents.timeline_agent._procedural") as mock:
        mock.get_procedures = AsyncMock(return_value=["Flag silent delays > 180 days"])
        yield mock

@pytest.fixture
def mock_episodic_store():
    with patch("agents.timeline_agent._episodic") as mock:
        mock.save_episode = AsyncMock()
        yield mock

@pytest.fixture
def base_state():
    return {
        "task": "Detect silent delays for NCT00000000",
        "nct_ids": ["NCT00000000"],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

@pytest.mark.asyncio
async def test_timeline_no_signals(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    mock_response = MagicMock(content="NO_SIGNALS_FOUND", tool_calls=[])
    mock_llm.ainvoke.return_value = mock_response
    
    result = await timeline_node(base_state)
    
    assert len(result["signals"]) == 0
    assert "timeline_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_timeline_detects_silent_delay(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests the chain: fetch details -> get amendments (empty) -> signal
    """
    
    # tool 1: fetch details (shows it is 2 years overdue)
    res1 = MagicMock(content="Fetching details...", tool_calls=[{"name": "fetch_study_details", "args": {"nct_id": "NCT00000000"}}])
    
    # tool 2: get amendments (returns empty list = silent delay)
    res2 = MagicMock(content="Checking amendments...", tool_calls=[{"name": "get_study_amendments", "args": {"nct_id": "NCT00000000"}}])
    
    # final signal
    sig = {"nct_id": "NCT00000000", "signal_type": "timeline_delay", "summary": "Silent delay of 24 months", "evidence": [], "confidence": 0.9}
    res3 = MagicMock(content=f"Found violation: {json.dumps(sig)}", tool_calls=[])

    mock_llm.ainvoke.side_effect = [res1, res2, res3]

    with patch("agents.timeline_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.side_effect = [
            '{"completion_date": "2021-01-01", "status": "COMPLETED"}',
            '{"amendments": []}'
        ]

        result = await timeline_node(base_state)

    assert len(result["signals"]) == 1
    assert result["signals"][0]["confidence"] == 0.9

@pytest.mark.asyncio
async def test_timeline_ignores_disclosed_delay(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests that an amendment prevents a signal from being generated
    """
    
    res1 = MagicMock(content="Checking...", tool_calls=[{"name": "get_study_amendments", "args": {"nct_id": "NCT00000000"}}])
    res2 = MagicMock(content="NO_SIGNALS_FOUND", tool_calls=[])

    mock_llm.ainvoke.side_effect = [res1, res2]

    with patch("agents.timeline_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        # mock tool returning a valid amendment (disclosed delay)
        mock_execute.return_value = '{"amendments": [{"date": "2022-01-01", "reason": "COVID-19"}]}'
        result = await timeline_node(base_state)

    assert len(result["signals"]) == 0

@pytest.mark.asyncio
async def test_timeline_error_handling(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    mock_llm.ainvoke.side_effect = Exception("Timeline LLM Error")
    result = await timeline_node(base_state)
    
    assert any("timeline_agent" in err for err in result["error_log"])