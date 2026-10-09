import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from agents.missing_results_agent import missing_results_node
from graph.state import GraphState

@pytest.mark.asyncio
async def test_missing_results_node_no_signals():
    """Test that the agent handles cases where no signals are found."""
    # Mock state
    state = {
        "task": "Find missing results for NCT123",
        "nct_ids": ["NCT123"],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

    # Mock LLM response to say NO_SIGNALS_FOUND
    with patch('agents.missing_results_agent._llm', new_callable=AsyncMock) as mock_llm:
        mock_response = MagicMock()
        mock_response.content = "NO_SIGNALS_FOUND"
        mock_response.tool_calls = []
        mock_llm.ainvoke.return_value = mock_response

        # Mock stores to avoid DB calls
        with patch('agents.missing_results_agent._procedural.get_procedures', new_callable=AsyncMock) as mock_proc, \
             patch('agents.missing_results_agent._episodic.save_episode', new_callable=AsyncMock) as mock_epi:

            mock_proc.return_value = ["Rule 1", "Rule 2"]

            result = await missing_results_node(state)

            assert "missing_results_agent" in result["agents_activated"]
            assert len(result["signals"]) == 0
            mock_epi.assert_called_once()

@pytest.mark.asyncio
async def test_missing_results_node_with_signal():
    """Test that the agent correctly parses and returns a signal."""
    state = {
        "task": "Find missing results for NCT123",
        "nct_ids": ["NCT123"],
        "signals": [],
        "agents_activated": [],
        "error_log": []
    }

    # Mock LLM response containing a JSON signal
    signal_json = '{"nct_id": "NCT123", "signal_type": "missing_results", "summary": "Missing results found", "evidence": ["fact1"], "confidence": 0.8}'

    with patch('agents.missing_results_agent._llm', new_callable=AsyncMock) as mock_llm:
        mock_response = MagicMock()
        mock_response.content = f"I found a signal: {signal_json}"
        mock_response.tool_calls = []
        mock_llm.ainvoke.return_value = mock_response

        with patch('agents.missing_results_agent._procedural.get_procedures', new_callable=AsyncMock) as mock_proc, \
             patch('agents.missing_results_agent._episodic.save_episode', new_callable=AsyncMock) as mock_epi:

            mock_proc.return_value = []

            result = await missing_results_node(state)

            assert len(result["signals"]) == 1
            assert result["signals"][0]["nct_id"] == "NCT123"
            assert result["signals"][0]["confidence"] == 0.8
            assert "missing_results_agent" in result["agents_activated"]
