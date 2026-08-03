# entry point for processing pipeline
# workflow in order :

# 1. Reads all cleaned study files from Google Cloud Storage bucket
# 2. Saves each study's metadata into the Cloud SQL studies table
#    (required first because chunks have a foreign key to studies)
# 3. Splits each study's text into overlapping chunks (chunker.py)
# 4. Converts each chunk into 1536 numbers via OpenAI (embedder.py)
# 5. Saves chunks + embeddings into Cloud SQL chunks table (vector_store.py)

# NOTE: chunks table has: nct_id REFERENCES studies(nct_id)
# this is a foreign key constraint (referential integrity)
# it means we CANNOT save a chunk for a study that does not yet exist in the studies table
# so we ALWAYS save study metadata to Cloud SQL BEFORE saving chunks
# getting this order wrong causes ForeignKeyViolationError on every insert

# sample output to expect :
# 2026-04-15 14:32:11 | INFO | Starting the processing pipeline!
# 2026-04-15 14:32:11 | INFO | Found 150 studies in GCS bucket!
# 2026-04-15 14:32:15 | INFO | Loaded 150 studies successfully!
# 2026-04-15 14:32:15 | INFO | Inserted 150 studies into database!
# 2026-04-15 14:32:16 | INFO | Chunked all studies! | total_chunks = 312
# 2026-04-15 14:32:16 | INFO | Starting embedding! | total_chunks = 312
# 2026-04-15 14:32:18 | INFO | Embedding batch 1/7 | chunks_in_batch = 50
# 2026-04-15 14:32:20 | INFO | Batch embedded successfully! | chunks = 50 | embedding_dims = 1536
# ... repeats for each batch ...
# 2026-04-15 14:32:45 | INFO | Chunks saved! | saved = 312 | skipped = 0
# 2026-04-15 14:32:45 | INFO | Processing completed!
# 2026-04-15 14:32:45 | INFO | Studies processed : 150
# 2026-04-15 14:32:45 | INFO | Chunks created    : 312
# 2026-04-15 14:32:45 | INFO | Chunks embedded   : 312
# 2026-04-15 14:32:45 | INFO | Chunks stored     : 312

import asyncio
from ingestion.document_parser import ParsedStudy
from ingestion.gcs_store import GCSStore
from processing.chunker import Chunker
from processing.embedder import Embedder
from processing.vector_store import VectorStore

from config.logging_config import setup_logging
logger = setup_logging(__name__)

async def run_processing():
    """
    runs / executes the entire processing pipeline from start to finish
    - reads cleaned studies from GCS bucket
    - chunks them
    - embeds them via OpenAI
    - stores everything in Cloud SQL for agent search & retrieval
    """
    gcs_store = GCSStore()
    chunker = Chunker()
    embedder = Embedder()

    logger.info("Loading parsed studies from GCS...")

    # ask GCS for a list of every NCT ID we saved during ingestion
    nct_ids = await gcs_store.list_processed_studies()

    logger.info(f"Found {len(nct_ids)} studies in GCS bucket!")

    studies : list[ParsedStudy] = []

    for nct_id in nct_ids:
        study = await gcs_store.load_parsed_study(nct_id)
        # loading the full ParsedStudy object for each NCT ID
        # this reads the "processed/studies/NCT*.json" file from GCS
        # and rebuilds it as a Pydantic ParsedStudy object
        if study:
            studies.append(study)
        
        logger.info(f"Loaded {len(studies)} studies successfully!")

    # step 1: opening the database connection
    async with VectorStore() as vector_store:
        # "async with" opens the connection pool when we enter
        # and guarantees it closes when we exit - even if an
        # error occurs halfway through this is the correct way to use VectorStore
        # never open it manually

        # step 2: save study metadata to cloud SQL
        logger.info("Saving study metadata to Cloud SQL...")

        studies_saved = 0
        for study in studies:
            success = await vector_store.save_study(
                study_data={
                    "nct_id":             study.nct_id,
                    "title":              study.title,
                    "sponsor":            study.sponsor,
                    "phase":              study.phase,
                    "status":             study.status,
                    "conditions":         study.conditions,
                    "interventions":      study.interventions,
                    "primary_outcome":    study.primary_outcome,
                    "secondary_outcomes": study.secondary_outcomes,
                    "start_date":         study.start_date,
                    "completion_date":    study.completion_date,
                    "results_posted":     study.results_posted,
                    "enrollment":         study.enrollment,
                    "gcs_path":           f"processed/studies/{study.nct_id}.json",
                }
                # save_study() accepts dict[str, Any], not a ParsedStudy object directly
            )

            if success:
                studies_saved += 1

        logger.info(
            f"Studies saved to Cloud SQL! | "
            f"saved = {studies_saved} | "
            f"total = {len(studies)}"
        )

        # step 3: chunk all studies
        logger.info("Chunking studies...")

        all_chunks = chunker.chunk_studies(studies)
        # chunk_studies() processes the entire list at once and
        # returns one flat list of TextChunk objects

        logger.info(f"Total chunks created: {len(all_chunks)}")

        # step 4: embed all chunks
        logger.info("Embedding chunks via OpenAI text-embedding-3-small...")

        embedded_chunks = await embedder.embed_chunks(all_chunks)
        # this is the step that costs OpenAI API credit
        # embed_chunks() sends chunks in batches of 50 to the OpenAI's embedding model
        # each chunk comes back with 1536 numbers attached
        # $0.02 per million tokens is the pricing of OpenAI text-embedding-3-small

        logger.info(f"Total chunks embedded: {len(embedded_chunks)}")

        # step 5: save chunks to cloud SQL
        logger.info("Saving embedded chunks to Cloud SQL...")

        chunks_stored = await vector_store.save_embedded_chunks(embedded_chunks)
        # save_embedded_chunks() inserts every EmbeddedChunk into the
        # chunks table - text, metadata, and the 1536 number vector
        # we use ON CONFLICT DO NOTHING on this chunks table so re-runs do not duplicate data

    # connection pool has now been closed 

    logger.info("*" * 50)                      # <- __aexit()__ of VectorStore runs here
    logger.info("Processing completed!")
    logger.info(f"Studies processed : {len(studies)}")
    logger.info(f"Chunks created    : {len(all_chunks)}")
    logger.info(f"Chunks embedded   : {len(embedded_chunks)}")
    logger.info(f"Chunks stored     : {chunks_stored}")
    logger.info("*" * 50)

if __name__ == "__main__":
    asyncio.run(run_processing())