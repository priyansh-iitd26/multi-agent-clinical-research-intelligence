# fetches research papers from PubMed that reference a specific clinical trial ID
# This is our second data source.
# What it does:
# Takes a NCT ID (e.g. NCT09664) (unique ID for every trial)
# searches PubMed for papers that reference that particular trial
# fetch the full abstract and metadata for each paper
# returns the raw paper records - no cleaning happens here

# why do we even need PubMed in the first place ?
# clinicaltrials.gov tells us what a study promised to measure
# PubMed tells us what researchers actually published in a particular study
# gap between these two things is where signals (output by agents) live

import asyncio
# asyncio.sleep() - pause between the requests to respect the rate limits
import httpx
# using httpx here (unlike ingestion/clinical_trials_client.py)
# because PubMed doesn't have same bot connection as clinicaltrials.gov
# httpx works fine with PubMed database - no 404 errors
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__) # __name__ = ingestion.pubmed_client

# constants
BASE_URL = settings.pubmed_base_url
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
FETCH_BATCH_SIZE = 20 # how many papers that I want to fetch in one efetch call (step-2)
RATE_LIMIT_SLEEP = 0.4 # how many seconds to sleep between API requests

# client class
class PubMedClient:
    """
    client for downloading research papers from pubmed
    works in 2 steps
    1. esearch - find paper IDs matching our query
    2. efetch - get full details (title, abstract, etc.) from paper IDs
    """
    def __init__(self):
        self._client : httpx.AsyncClient | None = None
        # unlike requests.Session, httpx.AsyncClient is natively async
        # we don't need asyncio.to_thread() here

    async def __aenter__(self) -> "PubMedClient":
        """
        called automatically when entering the "async with" block
        creates the http async client with shared configuration
        """
        self._client = httpx.AsyncClient(
            timeout = httpx.Timeout(REQUEST_TIMEOUT)
        )
        headers = {
            "Accept" : "application/json"
            # efetch returns xml regardless of the above header
            # we need to handle xml parsing separately
        }
        logger.info("PubMed client opened!")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        called automatically when exiting the "async with" block
        closes the httpx client and releases the network connections
        """
        # only close if the session was actually created
        if self._client:
            await self._client.aclose() # aclose() is async version of close() - must use await
            logger.info("PubMed client closed!")

    # core method : fetch papers for 1 trial
    
