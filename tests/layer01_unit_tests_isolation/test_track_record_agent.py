# unit testing track_record_agent in isolation

# 'macro' agent
    # doesn't look at one trial; it looks at the sponsor's global behavior
    # using `get_low_credibility_sponsors`

import pytest
import json
from unittest.mock import patch, AsyncMock, MagicMock

from graph.state import GraphState, SignalOutput
from agents.track_record_agent import track_record_node

@pytest.fixture
def mock_llm():
    with patch("agents.track_record_agent._llm") as mock:
        mock.ainvoke = AsyncMock()
        yield mock

@pytest.fixture
def mock_procedural_store():
    with patch("agents.track_record_agent._procedural") as mock:
        mock.get_procedures = AsyncMock(return_value=["Evaluate systemic patterns"])
        yield mock

@pytest.fixture
def mock_episodic_store():
    with patch("agents.track_record_agent._episodic") as mock:
        mock.save_episode = AsyncMock()
        yield mock

@pytest.fixture
def base_state():
    return {
        "task": "Evaluate sponsor credibility",
        "nct_ids": [],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

@pytest.mark.asyncio
async def test_track_record_no_low_credibility(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests path where all sponsors are above the credibility threshold
    """
    
    mock_response = MagicMock(content="NO_SIGNALS_FOUND", tool_calls=[])
    mock_llm.ainvoke.return_value = mock_response

    result = await track_record_node(base_state)
    
    assert len(result["signals"]) == 0
    assert "track_record_agent" in result["agents_activated"]

@pytest.mark.asyncio
async def test_track_record_detects_systemic_failure(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    """
    tests the loop: find low credibility -> get profile -> signal
    """
    
    # turn 1: find sponsors below 0.6
    res1 = MagicMock(content="Finding problematic sponsors...", tool_calls=[{"name": "get_low_credibility_sponsors", "args": {"threshold": 0.6}}])
    
    # turn 2: get specific profile for the worst one
    res2 = MagicMock(content="Getting profile for BadPharma...", tool_calls=[{"name": "get_sponsor_profile", "args": {"sponsor_name": "BadPharma"}}])
    
    # turn 3: signal
    sig = {"nct_id": "N/A", "signal_type": "low_credibility", "summary": "BadPharma: Credibility 0.3, 10 missing results", "evidence": ["10/12 studies missing results"], "confidence": 0.9}
    res3 = MagicMock(content=f"Pattern confirmed: {json.dumps(sig)}", tool_calls=[])

    mock_llm.ainvoke.side_effect = [res1, res2, res3]

    with patch("agents.track_record_agent._execute_tool", new_callable=AsyncMock) as mock_execute:
        mock_execute.side_effect = [
            '{"sponsors": [{"sponsor": "BadPharma", "credibility_score": 0.3}]}',
            '{"sponsor": "BadPharma", "total_studies": 12, "results_missing": 10}'
        ]

        result = await track_record_node(base_state)

    assert len(result["signals"]) == 1
    assert result["signals"][0]["confidence"] == 0.9

@pytest.mark.asyncio
async def test_track_record_malformed_json(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    mock_response = MagicMock(content="Pattern: { nct_id: 'N/A', confidence: 0.8 [BAD JSON]", tool_calls=[])
    mock_llm.ainvoke.return_value = mock_response

    result = await track_record_node(base_state)
    
    assert len(result["signals"]) == 0

@pytest.mark.asyncio
async def test_track_record_critical_error(base_state, mock_llm, mock_procedural_store, mock_episodic_store):
    mock_llm.ainvoke.side_effect = Exception("DB Connection Error!")
    result = await track_record_node(base_state)
    
    assert any("track_record_agent" in err for err in result["error_log"])