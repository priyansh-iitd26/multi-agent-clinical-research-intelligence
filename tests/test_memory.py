import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from memory.semantic_store import SemanticStore


@pytest.mark.asyncio
async def test_semantic_store_get_profile():
    """
    testing retrieving a sponsor profile
    """

    store = SemanticStore()

    # mocking the database pool and connection
    mock_pool = MagicMock()
    mock_conn = AsyncMock()

    # To mock 'async with pool.acquire() as conn', we need pool.acquire()
    # to return an object that is BOTH an awaitable (because of the async with)
    # and an async context manager

    class MockAcquireResult:

        def __await__(self):
            return self.__aenter__().__await__()

        async def __aenter__(self):
            return mock_conn

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_pool.acquire.return_value = MockAcquireResult()

    # mocking the fetchrow result
    mock_conn.fetchrow.return_value = {
        "sponsor": "Test Sponsor",
        "credibility_score": 0.8,
        "total_studies": 10,
        "results_posted": 8,
        "results_missing": 2,
        "broken_promises": 0,
        "avg_delay_days": 10.0,
        "last_updated": "2023-01-01"
    }

    with patch.object(store, '_ensure_pool', new_callable=AsyncMock), \
         patch.object(store, '_pool', mock_pool):

        profile = await store.get_sponsor_profile("Test Sponsor")

        assert profile is not None
        assert profile["sponsor"] == "Test Sponsor"
        assert profile["credibility_score"] == 0.8