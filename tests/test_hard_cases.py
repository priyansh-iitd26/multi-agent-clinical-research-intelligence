import pytest
from unittest.mock import AsyncMock, patch
from agents.missing_results_agent import missing_results_node
from graph.state import GraphState


@pytest.mark.asyncio
async def test_multi_signal_aggregation():
    """
    verifies that the state reducer correctly merges multiple signals
    """
    
    from graph.state import _merge_lists

    signals_a = [{"signal_type": "missing_results", "nct_id": "NCT001", "confidence": 0.9}]
    signals_b = [{"signal_type": "broken_promise", "nct_id": "NCT001", "confidence": 0.8}]

    merged = _merge_lists([], signals_a)
    merged = _merge_lists(merged, signals_b)

    assert len(merged) == 2
    assert any(s["signal_type"] == "missing_results" for s in merged)
    assert any(s["signal_type"] == "broken_promise" for s in merged)


@pytest.mark.asyncio
async def test_confidence_calibration_low_confidence():
    """
    verifies that low-confidence signals are preserved (not up-scored)
    """
    
    state = {
        "task": "Verify NCT002 (conflicting data)",
        "nct_id": ["NCT002"],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

    # Mock LLM to return a low-confidence signal
    noisy_response = '{"nct_id": "NCT002", "signal_type": "missing_results", "summary": "Maybe missing?", "confidence": 0.4}'

    with patch('agents.missing_results_agent._llm') as mock_llm:
        mock_response = AsyncMock()
        mock_response.content = noisy_response
        mock_response.tool_calls = []
        mock_llm.ainvoke.return_value = mock_response

        with patch('agents.missing_results_agent._procedural.get_procedures', new_callable=AsyncMock, return_value=[]), \
             patch('agents.missing_results_agent._episodic.save_episode', new_callable=AsyncMock):

            result = await missing_results_node(state)

            assert len(result["signals"]) == 1
            assert result["signals"][0]["confidence"] == 0.4
            assert result["signals"][0]["confidence"] < 0.6
