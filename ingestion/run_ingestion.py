#   This is the only file we actually run directly in ingestion/

#   This file can be thought like this analogy:
#   clinical_trials_client.py = "the person who goes shopping for ingredients"
#   pubmed_client.py = "the person who goes shopping at a different store"
#   document_parser.py = "the person who washes and chops everything"
#   gcs_store.py = "the person who puts everything in the fridge"
#   run_ingestion.py = "the head chef, who tells everyone what to do and in what order"

#   1. pick a list of medical conditions to search for (example: ["diabetes", "cancer"])
#   2. for each condition, ask ClinicalTrials.gov for matching studies
#   3. save the raw version of every study to GCS bucket
#   4. clean/parse each study using document_parser.py
#   5. save the cleaned study version to GCS bucket too

#   6. for each study, ask PubMed if any research papers mention it
#   7. save the raw version of every fetched paper to GCS bucket
#   8. clean/parse each paper using document_parser.py
#   9. save the cleaned paper version to GCS bucket too

#   10. print a final summary of everything that happened

#   a complete run with a handful of conditions usually takes a few
#   minutes — most of the time is spent waiting on the PubMed API,
#   since we deliberately slow down requests to respect their
#   rate limits (see RATE_LIMIT_SLEEP in pubmed_client.py)

import asyncio
# we need asyncio because our entire pipeline is built using async/await 
# asyncio.run() is the command that actually STARTS an async program 
# everything we built so far has been async functions waiting to be called
# this is where we finally call the very first one

# importing all four classes built earlier
# this file does not contain any new logic of its own for talking
# to APIs or saving files — it simply ORCHESTRATES these four
# classes, calling them in the right order
from ingestion.clinical_trials_client import ClinicalTrialsClient
from ingestion.pubmed_client import PubMedClient
from ingestion.document_parser import DocumentParser
from ingestion.gcs_store import GCSStore

from config.logging_config import setup_logging

logger = setup_logging(__name__)

# **************************************************************
# configuration - what to search for
# these are like control knobs for this run

# list of medical conditions we will search for
# for each condition in this list, we ask ClinicalTrials.gov
# for matching studies - starting small with just 2-3 conditions
# for testing
SEARCH_CONDITIONS = [
    "diabetes",
    "cardiovascular disease",
    "cancer"
]

# how many max studies to fetch for EACH condition above
# with 3 studies in SEARCH_CONDITIONS and 
# <= 50 MAX_STUDIES_PER_CONDITION, we get upto 150 studies total
MAX_STUDIES_PER_CONDITION = 20

# how many max PubMed papers to fetch per study
# most studies have ZERO papers referencing them
MAX_PAPERS_PER_STUDY = 10
# **************************************************************

# main pipeline function
async def run_ingestion():
    """
    runs the complete ingestion pipeline from start to finish

    downloads studies from ClinicalTrials.gov for every condition
    in SEARCH_CONDITIONS, saves them to GCS bucket (both raw and cleaned),
    then does the same for any related PubMed papers
    """
    logger.info("*" * 50)
    logger.info("Starting the ingestion pipeline!")
    logger.info(f"Conditions to search : {', '.join(SEARCH_CONDITIONS)}")
    logger.info(f"Max studies per condition : {MAX_STUDIES_PER_CONDITION}")
    logger.info("*" * 50)

    # create one instance of each helper class
    # we create one parser and one storage object
    # that gets reused for every single study &
    # paper in this run
    # no need to create a new one each time
    # these classes do not remember anything
    # between calls, so reusing them is both
    # safe as well as efficient
    parser = DocumentParser()
    gcs_store = GCSStore()

    total_studies = 0
    total_papers = 0

    # open both API clients
    async with ClinicalTrialsClient() as ct_client:
        async with PubMedClient() as pubmed_client:
            # "async with" opens BOTH clients and guarantees they
            # get closed properly afterwards — even if something
            # goes wrong halfway through we nest them because we need
            # BOTH of them open simultaneously throughout this run

            for condition in SEARCH_CONDITIONS:

                # step-1 : search ClinicalTrials.gov

                logger.info(f"Fetching studies for condition = {condition}")

                raw_studies = await ct_client.search_studies(
                    condition = condition,
                    max_results = MAX_STUDIES_PER_CONDITION
                )
                # raw studies is a list[dict[str, Any]]
                # above line is doing a LOT of work behind the scenes —
                # paginating through multiple pages,
                # retrying on network failures, applying our
                # custom headers to bypass bot protection
                # all of that complexity is abstracted inside the
                # ClinicalTrialsClient class

                logger.info(
                    f"Fetched {len(raw_studies)} studies! | "
                    f"condition = {condition}"
                )

                # step-2 : clean/parse each study we got

                # sent the whole raw_studies (list[dict[str, Any]]) to the parser
                parsed_studies = parser.parse_studies(raw_studies)

                # step-3 : save every study (raw & processed)

                # len(raw_studies) > len(parsed_studies) as
                # some raw studies might not be able to get
                # parsed (209-219 lines in document_parser.py)
                # we intend to store only those raw studies which were
                # successfully parsed
                # for that, we iterate over all parsed_studies and for
                # each parsed_study, we find it's corresponding raw study
                # match (based on NCT ID)
                # so, in this way we store only those raw studies which
                # have been parsed successfully

                for parsed_study in parsed_studies:
                    # next(..., None) returns the first match found
                    # or None if somehow nothing matched (not possible)
                    raw_study_match = next(
                        (r for r in raw_studies 
                        if r.get("protocolSection", {})
                        .get("identificationModule", {})
                        .get("nctId") == parsed_study.nct_id),
                        None
                    )

                    # saving the raw_study_match in GCS bucket
                    if raw_study_match:
                        await gcs_store.save_raw_study(
                            nct_id = parsed_study.nct_id,
                            raw_data = raw_study_match
                        )

                    # saving the cleaned/parsed version too
                    await gcs_store.save_parsed_study(parsed_study)

                    total_studies += 1

                    # step-4 : search PubMed for any papers for this particular parsed_study

                    # for THIS specific parsed_study, has anyone
                    # published a paper that references this trial?
                    # most of the time the answer will be "no papers
                    # found" (completely normal)
                    
                    raw_papers = await pubmed_client.fetch_papers_for_trial(
                        nct_id = parsed_study.nct_id,
                        max_results = MAX_PAPERS_PER_STUDY
                    )
                    # raw_papers is a list[dict[str, Any]]

                    # cleaning up whatever list of paper dicts come above
                    # (similar as done for studies above)

                    parsed_papers = parser.parse_papers(raw_papers)

                    # step-5 : save each paper (raw + parsed)
                    # similar to what done for studies

                    for parsed_paper in parsed_papers:

                        await gcs_store.save_raw_paper(
                            pmid = parsed_paper.pmid,
                            raw_data = parsed_paper.model_dump()
                        )

                        # NOTE: for papers, the "raw" version we save here
                        # is actually already somewhat cleaned dict from
                        # pubmed_client.py (already had to parse XML into a dict)
                        # it is still "raw" as compared to final ParsedPaper Pydantic object below

                        await gcs_store.save_parsed_paper(parsed_paper)

                        total_papers += 1

    # step-6: final summary

    # logging final summary after both above "async with" blocks have closed
    # both API clients have been safely shut down before we log final summary

    logger.info("*" * 50)
    logger.info("Ingestion completed!")
    logger.info(f"Studies saved : {total_studies}")
    logger.info(f"Papers saved : {total_papers}")
    logger.info("*" * 50)


if __name__ == "__main__":
    # __name__ is set to "__main__" only when this file is
    # directly run (python run_ingestion.py)

    # NOTE: if this file is ever imported by another file instead of run
    # directly, __name__ would be "ingestion.run_ingestion" instead,
    # and this block would be skipped - preventing the pipeline
    # from accidentally running just because some other file imported it

    # start the ingestion pipeline
    asyncio.run(run_ingestion())

# NOTE: we only call function objects with await when the function is defined using async def
# to call normal functions (defined using def keyword), we don't use await
# depends on whether the function performs asynchronous operations (I/O) or 
# just computes something in memory

# NOTE: Does this function have to wait for something outside the program?
# If yes, it is a good candidate for "async"
# examples: ClinicalTrialsClient, PubMedClient, and GCSStore use async def
# because they perform I/O operations (network requests to APIs or Google Cloud
# Storage), which involve waiting on external systems to respond
# DocumentParser doesn't need async def because it only performs
# in-memory processing (parsing dicts, extracting fields, and creating Pydantic objects)
# It doesn't wait on any external resource such as a network, API or database