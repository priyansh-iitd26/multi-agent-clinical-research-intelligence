# saves and loads the documents to and from the Google Cloud Storage (bucket)
# this is a permanent storage layer for all the raw and parsed docs
# saves the raw API responses to GCS bucket as JSON
# saves parsed study and paper records to GCS as JSON
# loads the docs back from GCS when the agent(s) need them
# lists available docs by prefix - useful for batch processing

# why we save the raw data first ?
# if parser has a bug, the raw originals are safe in GCS bucket
# we can re-parse them anytime without re-fetching from the API
# this is called raw zone - processed zone pattern typically used in data engineering

# folder structure expected in the GCS bucket
# raw/studies/NCT0089870.json ---> exactly what the API (ClinicalTrials.gov) returned
# raw/papers/9897948.json ---> exactly what PubMed returned
# processed/studies/NCT0089870.json ---> parsed study as JSON
# processed/papers/9897948.json ---> parsed paper as JSON

import json
import asyncio
from typing import Any
from google.cloud import storage
from config.settings import settings
from config.logging_config import setup_logging
from ingestion.document_parser import ParsedStudy, ParsedPaper

logger = setup_logging(__name__) # here __name__ = "ingestion.gcs_store"

# folder paths inside the gcs bucket
PREFIX_RAW_STUDIES = "raw/studies"
PREFIX_RAW_PAPERS = "raw/papers"
PREFIX_PROCESSED_STUDIES = "processed/studies"
PREFIX_PROCESSED_PAPERS = "processed/papers"

# gcs store class
class GCSStore:
    """
    handles saving data to and loading data from Google Cloud Storage
    
    IMPORTANT — WHY WE USE asyncio.to_thread() THROUGHOUT THIS FILE:

    Google's official storage library is "synchronous" — meaning
    when you ask it to upload a file, your whole program freezes
    and waits until the upload is done before doing anything else.

    But our entire system architecture is built to be "asynchronous" —
    meaning we want our program to be able to do OTHER things
    while waiting for slow operations like uploads to finish.

    asyncio.to_thread() is the bridge between these two worlds.
    It takes a synchronous function (like a GCS upload) and runs
    it in a separate background thread, while letting our main
    program keep working on other tasks in the meantime.
    """
    def __init__(self):
        # get an object to connect to the gcs
        self._client = storage.Client(project = settings.gcp_project_id)
        # get the reference to the specific bucket
        self._bucket = self._client.bucket(bucket_name = settings.gcs_bucket_name)

        logger.info(
            f"GCSStore initialized! | "
            f"bucket = {settings.gcs_bucket_name} | "
            f"project = {settings.gcp_project_id}"
        )

    # save raw study
    async def save_raw_study(
        self,
        nct_id : str,
        raw_data : dict[str, Any]
    ) -> str:
        """
        saves the exact, untouched API response for one study
        call this the moment we receive data from the API —
        before any cleaning or parsing happens

        args:
            nct_id: the study's unique NCT ID, used as the filename
            data:   the raw study dictionary to save

        returns:
            the path inside GCS where the file was saved
            Example: "raw/studies/NCT04788680.json"
        """
        target_gcs_path = f"{PREFIX_RAW_STUDIES}/{nct_id}.json"

        await self._upload_json(path=target_gcs_path, data=raw_data)

        logger.info(f"Saved raw study! | nct_id = {nct_id} | path = {target_gcs_path}")
        return target_gcs_path

    # save raw paper
    async def save_raw_paper(
        self,
        pmid: str,
        raw_data: dict[str, Any],
    ) -> str:
        """
        saves the exact, untouched API response for one PubMed paper
        same idea as save_raw_study — but for papers
        """

        target_gcs_path = f"{PREFIX_RAW_PAPERS}/{pmid}.json"

        await self._upload_json(path = target_gcs_path, data = raw_data)

        logger.info(f"Saved raw paper! | pmid = {pmid} | path = {target_gcs_path}")
        return target_gcs_path

    # save processed (parsed) study
    async def save_parsed_study(
        self,
        study : ParsedStudy
    ) -> str:
        """
        saves the cleaned version of a study — after document_parser.py
        has already processed it into a ParsedStudy type-validated object
        """
        target_gcs_path = f"{PREFIX_PROCESSED_STUDIES}/{study.nct_id}.json"

        await self._upload_json(path = target_gcs_path, data = study.model_dump())

        logger.info(
            f"Saved parsed study! | nct_id = {study.nct_id} | path = {target_gcs_path}"
        )
        return target_gcs_path
    
    # save processed (parsed) paper
    async def save_parsed_paper(
        self,
        paper : ParsedPaper
    ) -> str:
        """
        saves the cleaned version of a PubMed paper — after document_parser.py
        has already processed it into a ParsedPaper type-validated object
        """
        target_gcs_path = f"{PREFIX_PROCESSED_PAPERS}/{paper.pmid}.json"

        await self._upload_json(path = target_gcs_path, data = paper.model_dump())

        logger.info(
            f"Saved parsed study! | nct_id = {paper.pmid} | path = {target_gcs_path}"
        )
        return target_gcs_path

    # load a clean study from gcs
    async def load_parsed_study(
        self,
        nct_id : str
    ) -> ParsedStudy | None:
        """
        loads a previously saved, cleaned study back from GCS
        this is the reverse of save_parsed_study — we use this
        in the processing layer when we need to read studies
        back in to chunk and embed them

        args:
            nct_id: which study to load

        returns:
            a ParsedStudy object (if found)
            None if no study with that NCT ID exists in GCS
        """
        source_gcs_path = f"{PREFIX_PROCESSED_STUDIES}/{nct_id}.json"

        # download the json text and convert it back to python dict
        # for pydantic type-validation (private helper does this)
        fetched_study_dict = await self._download_json(path = source_gcs_path)

        if not fetched_study_dict:
            return None

        # rebuilding Pydantic object from python dict 
        # (unpacking into keyword args)
        return ParsedStudy(**fetched_study_dict)

    # list all studies we've already processed
    async def list_processed_studies(self) -> list[str]:
        """
        returns a list of every study's NCT ID currently saved
        in the "processed" folder of our bucket

        we use this later in the processing layer to know exactly
        which studies are available to chunk and embed, without
        needing to ask the database first

        returns:
            a list of NCT ID strings which are already processed
        """
        blobs = await asyncio.to_thread(
            self._bucket.list_blobs,
            prefix = PREFIX_PROCESSED_STUDIES
        )
        # "Blob" is GCS terminology for a single saved file
        # list_blobs() is synchronous, so we wrap it in asyncio.to_thread()
        # prefix means only show files whose path starts with "processed/studies"

        nct_ids = []
        for blob in blobs:
            # blob.name looks like "processed/studies/NCT997066.json"
            filename = blob.name.split("/")[-1]
            nct_id = filename.replace("/", "")

            if nct_id:
                nct_ids.append(nct_id)

        logger.info(f"Listed processed studies! | count = {len(nct_ids)}")
        return nct_ids

    # private helper: _upload_json()
    async def _upload_json(
        self,
        path : str, # where inside the bucket to save
        data : dict[str, Any] # python dict which we want to save
    ) -> None:
        """
        shared internal method that actually uploads into json format on GCS
        args:
            path: destination path inside the GCS bucket to save
            data: python dict which we want to save as json
        """
        json_bytes = json.dumps(data, indent=2, default=str).encode("utf-8")

        # create a reference to where this file will live in GCS
        # Doesn't upload anything yet — it is just a pointer to the destination
        blob = self._bucket.blob(path)

        # actually upload
        # upload_from_string is synchronous (it would normally freeze our program)
        # hence, we run it inside asyncio.to_thread()
        await asyncio.to_thread(
            blob.upload_from_string,
            json_bytes,
            content_type="application/json",
        )

    # private helper: _download_json()
    async def _download_json(
        self,
        path : str
    ) -> dict[str, Any] | None:
        """
        internal method that downloads a file from GCS
        and converts it back into a python dict
        """
        try:
            # pointer / point to the file we want to download
            blob = self._bucket.blob(path)

            # downloading the file's raw content as bytes
            # google library call is synchronous by default
            # hence, wrapping it inside asyncio.to_thread()
            json_bytes = await asyncio.to_thread(blob.download_as_bytes)

            return json.loads(json_bytes.decode("utf-8"))

        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                logger.warning(f"File not found in GCS | path={path}")
            else:
                logger.error(
                    f"Failed to download from GCS | path={path} | error={e}"
                )
            return None