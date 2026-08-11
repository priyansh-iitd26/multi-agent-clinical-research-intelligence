# this file defines tools that let agents fetch live research papers
# directly from PubMed during an analysis run

# why do we need PubMed tools separately from search_tools.py ?
# search_tools.py searches papers that are already in our database -
# papers we downloaded during ingestion and stored as chunks
# pubmed_tools.py fetches papers live from PubMed on demand
# the difference matters for the Side Effect Checker agent -
# it needs to find papers published after our ingestion pipeline run
# a paper published last week about a safety concern would not
# be in our database - but PubMed has it right now

# which agents use these tools ?
# Side Effect Checker -> fetch_papers_for_trial (primary user)
#                        compares official filings vs published papers
# Pattern Finder -> search_pubmed_by_query
#                   finds papers that discuss multiple trials
# Missing Results -> fetch_papers_for_trial
#                    checks if results were published in papers
#                    even though they were not posted officially

# important note on rate limits :
# PubMed allows 3 requests per second without an API key
# our PubMedClient already handles rate limiting with a 400ms sleep
# these tools respect that - do not call PubMed in rapid loops

import json
import asyncio
from langchain_core.tools import tool

from ingestion.pubmed_client import PubMedClient
# PubMedClient is the class we built in ingestion/ that handles
# all PubMed API communication - the two-step esearch -> efetch pattern,
# rate limiting, XML parsing, and retry logic
# we re-use it here rather than re-implementing PubMed API calls

from ingestion.document_parser import DocumentParser
# DocumentParser cleans raw PubMed responses into ParsedPaper objects

from config.logging_config import setup_logging

logger = setup_logging(__name__)

# shared instances

# one DocumentParser shared across all tool calls
# DocumentParser is stateless - safe to re-use many times
# created once at module level (outside all functions) to avoid repeated object creation
# PubMedClient is NOT created here - it is an async context manager
# it manages resources like HTTP sessions, network connections, authentication, cleanup
# objects with __aenter__ and __aexit__ are context managers
# (used with "async with") so we create it fresh inside each tool call

_parser = DocumentParser()


def _run_async(coroutine):
    """
    runs an async coroutine from synchronous code

    - coroutine
    calling an async function does NOT execute it immediately

        data = store.search(...)          # coroutine object
        data = await store.search(...)    # executes and returns result

    - a coroutine is simply a suspended computation waiting to be
    executed by python's asyncio event loop

    this helper takes such a coroutine, runs it to completion,
    waits for the result, and returns it to synchronous code

    args:
        coroutine: an unawaited coroutine returned by an async function

    returns:
        the value produced by the async function
    """

    # retrieve the current asyncio event loop
    # the event loop is responsible for scheduling and executing
    # asynchronous tasks such as database queries and network requests
    loop = asyncio.get_event_loop()

    # execute the coroutine until it completes, block the current thread
    # while it runs, and return its result
    return loop.run_until_complete(coroutine)


# TOOL 1: fetch_papers_for_trial
@tool
def fetch_papers_for_trial(
    nct_id: str,
    max_papers: int = 10,
) -> str:
    """
    fetch all published research papers that reference a specific
    clinical trial, LIVE from PubMed right now

    this is the primary tool for the Side Effect Checker agent
    it finds papers where authors reported their findings about
    a specific trial - then the agent compares those findings
    against what the official trial filing says

    WHY THIS IS POWERFUL:
    Official filings are written by the sponsor.
    Published papers are written by independent researchers.
    When these two sources DISAGREE about safety or outcomes,
    that disagreement is a signal worth investigating.

    Example scenario:
    Official filing says: "No serious adverse events observed"
    Published paper says: "Three patients were hospitalised"
    → Side Effect Checker flags this as a safety gap signal.

    Args:
        nct_id:     The trial's NCT ID to search for in PubMed.
                    Example: "NCT04788680"
        max_papers: Maximum papers to fetch. Default 10.
                    Keep low — each paper adds to agent context window.
                    More than 15 papers can overwhelm the agent.

    Returns:
        JSON string with all papers found, including:
        - title, abstract, journal, authors, publication date
        - word_count (very short abstracts may indicate limited detail)
        Empty list if no papers reference this trial in PubMed.
    """

    logger.info(
        f"Tool called: fetch_papers_for_trial | "
        f"nct_id = {nct_id} | max_papers = {max_papers}"
    )

    async def _fetch():
        # inner async function - contains all the async logic
        # we define it here and run it with _run_async() below
        # this pattern keeps async code cleanly separated from
        # the synchronous tool interface that LangGraph expects

        async with PubMedClient() as client:

            papers = await client.fetch_papers_for_trial(
                nct_id = nct_id,
                max_results = max_papers
            )

            return papers
            # papers is a list of raw paper dictionaries
            # each dict has: pmid, title, abstract, journal,
            # pub_date, authors, nct_ids_referenced

    try:
        raw_papers = _run_async(_fetch())
        # run the async fetch synchronously
        # raw_papers is a list of raw dicts from PubMed

        if not raw_papers:
            # PubMed has no papers referencing this trial
            # this is common - most trials are never directly cited
            # the agent reads this and notes "no published research found."
            return json.dumps({
                "nct_id":  nct_id,
                "papers":  [],
                "count":   0,
                "message": f"No published papers found on PubMed that "
                           f"reference trial {nct_id}. The trial may not "
                           "have published results in academic journals, "
                           "or results may only exist as grey literature.",
            }, indent = 2)

        parsed_papers = _parser.parse_papers(raw_papers = raw_papers)

        papers_list = []

        for paper in parsed_papers:

            paper_dict = paper.model_dump()

            papers_list.append({
                "pmid": paper_dict["pmid"],
                "title": paper_dict["title"],
                "abstract": paper_dict["abstract"],
                "journal": paper_dict["journal"],
                "pub_date": paper_dict["pub_date"],
                "authors": paper_dict["authors"][:5],
                "word_count": paper_dict["word_count"],
                "nct_ids_referenced": paper_dict["nct_ids_referenced"],
            })

        return json.dumps({
            "nct_id": nct_id,
            "papers": papers_list,
            "count": len(papers_list),
        }, indent = 2, default = str)

    except Exception as e:
        logger.error(
            f"fetch_papers_for_trial failed! | "
            f"nct_id = {nct_id} | error = {e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error":  str(e),
            "papers": [],
            "count":  0,
        })


# TOOL 2: search_pubmed_by_query
@tool
def search_pubmed_by_query(
    query: str,
    max_papers: int = 5,
) -> str:
    """
    search PubMed with a free-text query and fetch matching papers

    Use this tool when you want to find papers about a topic,
    drug, or condition — not just papers about one specific trial.

    Different from fetch_papers_for_trial which searches by NCT ID.
    This tool accepts any PubMed search query.

    Examples:
    - "semaglutide cardiovascular outcomes 2023"
    - "metformin diabetes safety adverse events"
    - "Novo Nordisk clinical trial results transparency"

    The Pattern Finder agent uses this to find papers that discuss
    multiple trials from the same sponsor — revealing systemic patterns.

    Args:
        query:      Any PubMed-compatible search query.
                    Can include drug names, conditions, author names,
                    journal names, or any combination.
        max_papers: Maximum papers to return. Default 5.
                    Keep low — each paper adds to agent context.

    Returns:
        JSON string with matching papers from PubMed.
    """

    logger.info(
        f"Tool called: search_pubmed_by_query | "
        f"query = '{query[:60]}' | max_papers = {max_papers}"
    )

    async def _search():
        # inner async function for the PubMed API call
        # two steps as always with PubMed:
        # step 1: esearch -> get paper IDs matching the query
        # step 2: efetch -> get full paper details for those IDs

        async with PubMedClient() as client:

            paper_ids = await client._search_paper_ids(
                nct_id = query,
                # we reuse _search_paper_ids() here with the query
                # as the search term instead of an NCT ID
                # PubMed's esearch accepts any search string -
                # not just NCT IDs
                # note: we are calling a "private" method (underscore prefix)
                # this is acceptable here because we are in the tools layer
                # which is a peer module - not external code
                # in a stricter design, PubMedClient would have a
                # public search_by_query() method, for this build,
                # reusing _search_paper_ids() avoids code duplication

                max_results = max_papers
            )

            if not paper_ids:
                return []

            papers = await client._fetch_paper_details(
                paper_ids = paper_ids
            )

            return papers

    try:
        raw_papers = _run_async(_search())

        if not raw_papers:
            return json.dumps({
                "query":   query,
                "papers":  [],
                "count":   0,
                "message": f"No papers found on PubMed for query: '{query}'. "
                           "Try a broader search term or different keywords.",
            }, indent = 2)

        parsed_papers = _parser.parse_papers(raw_papers = raw_papers)

        papers_list = []

        for paper in parsed_papers:
            paper_dict = paper.model_dump()

            papers_list.append({
                "pmid": paper_dict["pmid"],
                "title": paper_dict["title"],
                "abstract": paper_dict["abstract"],
                "journal": paper_dict["journal"],
                "pub_date": paper_dict["pub_date"],
                "authors": paper_dict["authors"][:5],
                "word_count": paper_dict["word_count"],
                "nct_ids_referenced": paper_dict["nct_ids_referenced"]
            })

        return json.dumps({
            "query":  query,
            "papers": papers_list,
            "count":  len(papers_list),
        }, indent = 2, default = str)

    except Exception as e:
        logger.error(
            f"search_pubmed_by_query failed! | "
            f"query = {query} | error = {e}"
        )
        return json.dumps({
            "query":  query,
            "error":  str(e),
            "papers": [],
            "count":  0
        })


# TOOL 3: compare_filing_vs_papers
@tool
def compare_filing_vs_papers(
    nct_id: str,
    filing_summary: str
) -> str:
    """
    Fetch papers for a trial and compare them against the official filing.

    This is the Side Effect Checker's most powerful tool.
    It fetches all published papers about a trial, then returns
    both the official filing summary AND the papers — so the agent
    can compare them and identify discrepancies.

    The agent looks for:
    - Safety events mentioned in papers but not in the filing
    - Different severity levels (filing says "mild", paper says "serious")
    - Outcomes reported in papers that differ from the primary outcome
    - Results published in papers when no official results were posted

    Args:
        nct_id:          The trial's NCT ID.
        filing_summary:  A summary of what the official filing says.
                         The agent provides this from earlier tool calls.
                         Example: "Filing reports no serious adverse events.
                                   Primary outcome was HbA1c reduction.
                                   Results posted: No."

    Returns:
        JSON string with:
        - filing_summary: what the agent passed in (echoed back)
        - papers: all published papers found on PubMed
        - comparison_note: guidance on what to look for
        - papers_count: how many papers were found
    """

    logger.info(
        f"Tool called: compare_filing_vs_papers | nct_id = {nct_id}"
    )

    async def _fetch():
        async with PubMedClient() as client:
            papers = await client.fetch_papers_for_trial(
                nct_id = nct_id,
                max_results = 15
            )
            return papers

    try:
        raw_papers = _run_async(_fetch())
        parsed_papers = _parser.parse_papers(raw_papers = raw_papers or [])

        papers_list = []
        for paper in parsed_papers:
            paper_dict = paper.model_dump()
            papers_list.append({
                "pmid": paper_dict["pmid"],
                "title": paper_dict["title"],
                "abstract": paper_dict["abstract"],
                "journal": paper_dict["journal"],
                "pub_date": paper_dict["pub_date"],
                "authors": paper_dict["authors"][:3]
            })

        comparison_note = (
            "Compare the filing_summary above against each paper's abstract. "
            "Look specifically for: "
            "(1) adverse events mentioned in papers but absent from filing, "
            "(2) different severity descriptions for the same event, "
            "(3) outcome results that contradict the filing's claims, "
            "(4) results data in papers when filing shows results_posted = False."
        )
        # this note is guidance for GPT-4o - it tells the agent
        # exactly what to look for when comparing the two sources
        # clear instructions in tool output = better agent reasoning
        # Without this note, the agent might just summarise the papers
        # instead of actively looking for discrepancies

        return json.dumps({
            "nct_id": nct_id,
            "filing_summary": filing_summary,
            "papers": papers_list,
            "papers_count": len(papers_list),
            "comparison_note": comparison_note,
            "has_papers": len(papers_list) > 0
        }, indent = 2, default = str)

    except Exception as e:
        logger.error(
            f"compare_filing_vs_papers failed! | "
            f"nct_id = {nct_id} | error = {e}"
        )
        return json.dumps({
            "nct_id": nct_id,
            "error": str(e),
            "papers": []
        })


# tools registry

# list of all PubMed tools defined in this file
# exported for graph_builder.py to assign to specific agents

# which agents use which PubMed tools:
# side_effect_agent -> compare_filing_vs_papers (primary)
#                      fetch_papers_for_trial
# pattern_finder_agent -> search_pubmed_by_query
#                         fetch_papers_for_trial
# missing_results_agent -> fetch_papers_for_trial
#                          (checks if results exist in papers
#                           even when not officially posted)

ALL_PUBMED_TOOLS = [
    fetch_papers_for_trial,
    search_pubmed_by_query,
    compare_filing_vs_papers
]