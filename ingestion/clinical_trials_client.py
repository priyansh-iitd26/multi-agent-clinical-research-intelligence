# it is entry point to all the data
# fetches all the clinical study records from clinicaltrials.gov
# connect to the clinicaltrials.gov public API
# search for the studies by condition, intervention or sponsor
# fetches the full study details for each result
# handles pagination - returns 100 results per page
# handles rate limiting and retries automatically instead of crashing
# async - concurrency method
# returns raw study data exactly as the API gave it to us
# we never modify the data here - that is document_parser.py's job

import asyncio
import requests
from typing import Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

BASE_URL = settings.clinical_trials_base_url
PAGE_SIZE = settings.clinical_trials_page_size
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/139.0.0.0 Safari/537.36"
    ),
}

# client class
# async with pattern

class ClinicalTrialsClient:
    """
    this client handles everything needed to talk to the API
    opening and closing the HTTP session
    handling pagination
    retrying automatically when the network fails
    """

    def __init__(self):
        self._session : requests.Session | None = None

    async def __aenter__(self) -> "ClinicalTrialsClient":
        """
        called automatically when we enter aync with block
        creates the HTTP session and sets the shared headers
        """
        self._session = requests.Session()
        self._session.headers.update(HEADERS)
        logger.info("ClinicalTrials client opened!")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        this method is called automatically when we exit the "async with" block
        closes the HTTP session and releases the TCP connection
        runs even if error occurs within the async block
        """
        if self._session:
            # only close if the session was actually created
            self._session.close()
            logger.info("ClinicalTrials client closed!")

    async def search_studies(
        self, 
        condition : str | None = None,
        intervention : str | None = None,
        sponsor : str | None = None,
        status : list[str] | None = None,
        max_results : int = 100,
    ) -> list[dict[str, Any]]:
        """
        searches clinicaltrials.gov and returns a list of study records
        dict : returns : study fields - nct_id, title, sponsor, outcomes, dates, etc.
        """
        all_studies : list[dict[str, Any]] = []
        next_page_token : str | None = None
        page_number = 0

        logger.info(
            f"Searching studies | "
            f"condition = {condition}"
            f"intervention = {intervention}"
            f"sponsor = {sponsor}"
            f"max_results = {max_results}" 
        )

        # pagination loop
        while len(all_studies) < max_results:
            page_number += 1
            params = self._build_search_params(
                condition = condition,
                intervention = intervention,
                sponsor = sponsor,
                status = status,
                page_token = next_page_token
            )
            # _build_search_params is a private helper which
            # returns : dict : {"page_size": 100, "format": "json", "query.cond": "diabetes"}

            response_data = await self._fetch_page(params = params)
            # _fetch_page is another private helper which returns a JSON response as python dict

            if not response_data:
                break

            page_studies = response_data.get("studies", [])

            if not page_studies:
                logger.info("No more studies available - pagination complete!")
                break

            all_studies.extend(page_studies)

            logger.info(
                f"Page - {page_number}"
                f"fetched = {len(page_studies)}"
                f"total so far = {len(all_studies)}"
            )

            next_page_token = response_data.get("nextPageToken")

            if not next_page_token:
                logger.info("Last page reached - no nextPageToken in response!")
                break

        all_studies = all_studies[:max_results]

        logger.info(
            f"Search complete | "
            f"total studies returned = {len(all_studies)}"
        )

        return all_studies

    # fetch study by ID
    async def fetch_study(self, nct_id : str) -> dict[str, Any] | None:
        """
        fetches the complete record for 1 specific study (nct_id)
        nct_id is unique identifier for every study on clinicaltrials.gov
        returns : dict with all study (nct_id) details if that study is found
        if study is not found, it returns None 
        """
        logger.info(f"Fetching study | nct_id = {nct_id}")

        # internal helper function
        def _get_study():
            # inner function : makes the actual HTTP GET request
            # regular non-async function as request is synchronous
            return self._session.get(
                f"{BASE_URL}/studies/{nct_id}",
                timeout = REQUEST_TIMEOUT,
            )

        try:
            # asyncio.to_thread() runs _get_study() in a background thread
            response = await asyncio.to_thread(_get_study)
            response.raise_for_status()

            return response.json()

        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"Study not found | "
                f"nct_id = {nct_id}"
                f"status = {e.response.status_code}"
            )
            return None

        except Exception as e:
            logger.error(
                f"Failed to fetch study | "
                f"nct_id = {nct_id}"
                f"error = {e}"
            )
            return None

    def _build_search_params(
            self,
            condition : str | None,
            intervention : str | None,
            sponsor : str | None,
            status : list[str] | None,
            page_token : str | None,
    ) -> dict[str, Any]:
        """
        build the query parameter dictionary for one API request
        returns a dict of query params ready to be sent to API
        """
        params : dict[str, Any] = {
            "pageSize" : PAGE_SIZE,
            "format" : "json",
        }

        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        if sponsor:
            params["query.spons"] = sponsor
        if status:
            params["filter.overallStatus"] = "|".join(status)
        if page_token:
            params["pageToken"] = page_token

        return params

    # retry decorator from tenacity
    @retry(
        stop = stop_after_attempt(MAX_RETRIES),
        wait = wait_exponential(multiplier = 1, min = 1, max = 8),
        retry = retry_if_exception_type(
            (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
        )
    )

    async def _fetch_page(self, params:dict[str, Any]) -> dict[str, Any] | None:
        """
        this makes one GET request to /studies endpoint
        """
        def _get():
            # inner function makes the actual HTTP request
            return self._session.get(
                f"{BASE_URL}/studies",
                params = params,
                timeout = REQUEST_TIMEOUT,
            )

        try:
            response = await asyncio.to_thread(_get)
            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            logger.warning(
                f"Request timed out after {REQUEST_TIMEOUT} seconds! Retrying..."
            )
            raise

        except requests.exceptions.ConnectionError:
            logger.warning(
                f"Connection Error! Retrying..."
            )
            raise

        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP error from API | "
                f"status = {e.response.status_code}"
                f"url = {e.response.url}"
            )
            # return None for HTTP errors — do not retry these
            return None

        except Exception as e:
            logger.error(
                f"Unexpected error fetching page | "
                f"error = {e}"
            )
            return None