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

    # core method - fetch papers for 1 trial
    async def fetch_papers_for_trial(
        self,
        nct_id : str,
        max_results : int = 50
    ) -> list[dict[str, Any]]:
        """
        fetches all PubMed papers that reference a specific clinical trial.

        handles the two-step process internally:
        step 1: esearch - find paper IDs for this NCT ID
        step 2: efetch  - get full details for those paper IDs

        returns :
            list of paper dictionaries with title, abstract, authors etc.
            empty list if no papers found or request failed.
        """
        logger.info(f"Fetching PubMed papers | nct_id={nct_id}")

        paper_ids = await self._search_paper_ids(
            nct_id=nct_id,
            max_results=max_results,
        )

        if not paper_ids:
            logger.info(f"No PubMed papers found | nct_id={nct_id}")
            return []

        logger.info(f"Found {len(paper_ids)} paper IDs | nct_id={nct_id}")

        papers = await self._fetch_paper_details(paper_ids=paper_ids)

        logger.info(
            f"PubMed fetch complete | "
            f"nct_id={nct_id} | "
            f"papers_returned={len(papers)}"
        )

        return papers

    # core method - fetch papers for multiple trials
    async def fetch_papers_for_trials(
        self,
        nct_ids : list[str],
        max_per_trial : int = 20
    ) -> dict[str, list[dict[str, Any]]]:
        """
        fetches papers for clinical trials
        processes trials one at a time with a sleep between each as per rate limits
        returns : dictionary mapping each NCT ID to its list of papers
        {
            "NCT087908" : [{"pmid": "123", "title": "", "authors": ""}, {}]
        }
        """
        results : dict[str, list[dict[str, Any]]] = {}
        # keys : NCT IDs
        # values : list of dictionary of papers corresponding to each NCT ID

        for i, nct_id in enumerate(nct_ids):

            logger.info(
                f"Processing trial {i + 1}/{len(nct_ids)} | "
                f"nct_id = {nct_id}"
            )

            papers = await self.fetch_papers_for_trial(
                nct_id = nct_id,
                max_results = max_per_trial
            )

            results[nct_id] = papers

            if i < len(nct_ids) - 1:
                await asyncio.sleep(RATE_LIMIT_SLEEP)

        return results

    # private method - search for paper IDs (esearch)
    @retry(
        stop = stop_after_attempt(MAX_RETRIES),
        wait = wait_exponential(multiplier = 1, min = 1, max = 8),
        retry = retry_if_exception_type(
            (httpx.TimeoutException,httpx.ConnectError)
        )
    )

    async def _search_paper_ids(
        self,
        nct_id : str,
        max_results : int
    ) -> list[str]:
        """
        calls the PubMed esearch to get the paper IDs for a given NCT ID
        returns : list of PubMed paper ID strings
        empty list if no papers found or request failed.
        """
        try:
            assert self._client is not None
            response = await self._client.get(
                f"{BASE_URL}/esearch.fcgi",
                params = {
                    "db" : "pubmed",
                    "term" : f"{nct_id}[si]",
                    "retmax": max_results,
                    "retmode": "json",
                    "usehistory": "n"
                }
            )

            response.raise_for_status()

            data = response.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            await asyncio.sleep(RATE_LIMIT_SLEEP)

            return id_list

        except httpx.TimeoutException:
            logger.warning(
                f"Timeout searching PubMed | nct_id = {nct_id} retrying..."
            )
            raise

        except httpx.ConnectError:
            logger.warning(
                f"Connection Error, Searching PubMed | nct_id = {nct_id} retrying..."
            )
            raise

        except Exception as e:
            logger.error(
                f"Failed to search PubMed | nct_id = {nct_id}"
                f"Error = {e}"
            )
            return []

    # private method - fetch paper details(efetch)
    async def _fetch_paper_details(
        self,
        paper_ids : list[str]
    ) -> list[dict[str, Any]]:
        """
        calls PubMed's efetch
        gets the full paper details for given list of paper IDs
        FETCH_BATCH_SIZE(20)
        sleeps between the batches to respect PubMed's rate limits
        """
        all_papers : list[dict[str, Any]] = []

        batches = [
            paper_ids[i: i + FETCH_BATCH_SIZE]
            for i in range(0, len(paper_ids), FETCH_BATCH_SIZE)
        ]

        for batch_num, batch in enumerate(batches):
            logger.info(
                f"Fetching paper details | "
                f"batch = {batch_num + 1}/{len(batches)}"
                f"total papers in current batch = {len(batch)}"
            )

            batch_papers = await self._fetch_batch(paper_ids = batch)

            all_papers.extend(batch_papers)

            if batch_num < len(batches) - 1:
                await asyncio.sleep(RATE_LIMIT_SLEEP)

        return all_papers
       
    # private method - fetch batch
    @retry(
        stop = stop_after_attempt(MAX_RETRIES),
        wait = wait_exponential(multiplier = 1, min = 1, max = 8),
        retry = retry_if_exception_type(
            (httpx.TimeoutException,httpx.ConnectError)
        )
    )

    async def _fetch_batch(
        self,
        paper_ids : list[str]
    ) -> list[dict[str, Any]]:
        """
        fetches full details for one batch of paper IDs using efetch
        efetch returns XML - parse it into Python dictionaries here

        returns: list of parsed paper dictionaries for this batch
        """
        try:
            assert self._client is not None
            response = await self._client.get(
                f"{BASE_URL}/efetch.fcgi",
                # the efetch endpoint
                # full url: https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
                params = {
                    "db" : "pubmed",
                    "id" : ",".join(paper_ids),
                    "retmode" : "xml",
                    "rettype" : "abstract"
                    # "rettype" : "abstract" gives the level of depth to extract from each paper
                }
            )

            response.raise_for_status()

            papers = self._parse_xml_response(xml_text = response.text)

            await asyncio.sleep(RATE_LIMIT_SLEEP)

            return papers

        except httpx.TimeoutException:
            logger.warning(
                f"Timeout fetching paper batch! Retrying..."
            )
            raise

        except httpx.ConnectError:
            logger.warning(
                f"Connection error fetching paper batch! Retrying..."
            )
            raise

        except Exception as e:
            logger.error(
                f"Failed to fetch the paper batch!"
                f"Error = {e}"
            )
            return []

    # private method - parses raw XML response
    def _parse_xml_response(self, xml_text) -> list[dict[str, Any]]:
        """
        parses the XML response from PubMed efetch into a list of
        clean Python dictionaries
        args:
            xml_text: The raw XML string from the efetch response
        returns:
            list of paper dictionaries with standardised fields
        """
        import xml.etree.ElementTree as ET

        papers : list[dict[str, Any]] = []

        try:
            root = ET.fromstring(xml_text)

            for article in root.findall(".//PubmedArticle"):
                paper = self._extract_paper_fields(article)

            if paper:
                papers.append(paper)

        except ET.ParseError as e:
            logger.error(
                f"Failed to parse PubMed XML response"
                f"Error = {e}"
            )

        return papers

    def _extract_paper_fields(self, article_element: Any) -> dict[str, Any] | None:
        """
        extracts the fields we need from one PubmedArticle XML element
        args:
            article_element: one PubmedArticle XML element
        returns:
            dict with the paper's key fields
            None if extraction failed completely
        """

        import xml.etree.ElementTree as ET

        def get_text(element: Any, path: str, default: str = "") -> str:
            node = element.find(path)
            return node.text.strip() if node is not None and node.text else default

        try:
            pmid  = get_text(article_element, ".//PMID")
            title = get_text(article_element, ".//ArticleTitle")

            abstract_texts = article_element.findall(".//AbstractText")
            abstract = " ".join(
                node.text.strip()
                for node in abstract_texts
                if node.text
            )

            pub_year  = get_text(article_element, ".//PubDate/Year")
            pub_month = get_text(article_element, ".//PubDate/Month", "01")
            pub_date  = f"{pub_year}-{pub_month}" if pub_year else ""

            journal = get_text(article_element, ".//Journal/Title")

            author_elements = article_element.findall(".//Author")
            authors = []
            for author in author_elements:
                last  = get_text(author, "LastName")
                first = get_text(author, "ForeName")
                if last:
                    authors.append(f"{last}, {first}".strip(", "))

            nct_ids_referenced = [
                id_elem.text.strip()
                for id_elem in article_element.findall(
                    ".//DataBankList/DataBank/AccessionNumberList/AccessionNumber"
                )
                if id_elem.text and id_elem.text.strip().startswith("NCT")
            ]

            return {
                "pmid":               pmid,
                "title":              title,
                "abstract":           abstract,
                "journal":            journal,
                "pub_date":           pub_date,
                "authors":            authors,
                "nct_ids_referenced": nct_ids_referenced,
                "source":             "pubmed",
            }

        except Exception as e:
            logger.error(
                f"Failed to extract paper fields | pmid = UNKNOWN | error={e}"
            )
            return None