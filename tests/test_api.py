import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from api.routers.analysis import run_analysis
from api.schemas import AnalysisRequest

@pytest.mark.asyncio
async def test_run_analysis_success():
    """Test the analysis endpoint returns a valid response."""
    request = AnalysisRequest(task="Test Task", nct_ids=["NCT123"], max_studies=5)

    # Mock Dependencies
    mock_graph = AsyncMock()
    mock_graph.ainvoke.return_value = {
        "signals": [{"nct_id": "NCT123", "agent": "test_agent", "confidence": 0.9, "summary": "test", "signal_type": "test"}],
        "agents_activated": ["test_agent"],
        "final_brief": "Test Brief"
    }

    mock_hitl = AsyncMock()
    mock_hitl.process_signal.return_value = {"action": "saved_directly"}

    with patch('api.routers.analysis.get_graph', return_value=mock_graph), \
         patch('api.routers.analysis.get_hitl', return_value=mock_hitl):

        # Since run_analysis uses Depends(), we call it directly and pass dependencies
        response = await run_analysis(request, hitl=mock_hitl, graph=mock_graph)

        assert response.run_id is not None
        assert response.total_signals == 1
        assert response.final_brief == "Test Brief"
        assert "test_agent" in response.agents_activated
