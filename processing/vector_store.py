# this file saves EmbeddedChunks into Cloud SQL's chunks table
# and provides semantic search (finding chunks by MEANING
# rather than by keyword)

# this is the HEART of the entire system
# every agent "query" flows through this file
# when the Missing Results agent asks:
# "find studies where sponsor never posted results"
# this file is what finds the answer

# 1. the agent's query gets converted to 1536 numbers
# 2. this file sends those 1536 numbers to Cloud SQL
# 3. pgvector compares them against every chunk's 1536 numbers
# 4. the chunks whose numbers are CLOSEST get returned
# 5. "closest" is measured using cosine similarity - the <=> operator

# why asyncpg and not SQLAlchemy ?
# SQLAlchemy is great for standard queries but pgvector's
# vector type is not natively supported by SQLAlchemy's ORM
# asyncpg is a raw async PostgreSQL driver it gives us complete control 
# over the SQL we write, which means we can use pgvector's custom operators 
# (<=> for cosine distance) without any compatibility issues

# NOTE: the codec
# asyncpg does not know what a vector type is by default
# PostgreSQL knows, but asyncpg needs to be taught how to
# convert between Python lists and PostgreSQL vector columns
# we register a custom codec - a translator - that handles this
# without it, every insert and select would crash with a type error


import asyncio
import asyncpg # fastest async PostgreSQL driver for Python
import json
from typing import Any
from processing.embedder import EmbeddedChunk
# EmbeddedChunk is our input (chunks with embeddings attached)

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

# configuration

# min number of database connections to keep open at all times
# we always keep atleast 2 ready to handle any requests
POOL_MIN_SIZE = 2

# max number of database connections allowed at the same time
# more connections => more parallelism => but more db memory usage
# 10 is a safe ceiling for our db-f1-micro Cloud SQL instance
POOL_MAX_SIZE = 10

# default number of similar chunks to return per search query
# when an agent searches, it gets back the 5 most relevant chunks
# by default, agents can override this if they need more
TOP_K_DEFAULT = 5

# vector store class
class VectorStore:
    """
    saves EmbeddedChunks to Cloud SQL and enables semantic search
    over them using pgvector's cosine similarity operator

    lifecycle:
        async with VectorStore() as vs:
            await vs.save_embedded_chunks(chunks)
            results = await vs.search(query_embedding)
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        # connection pool
        # starts as None - created when initialize() is called

    # async context manager support
    async def __aenter__(self) -> "VectorStore":
        """
        Allows using VectorStore with "async with" pattern
        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Closes the connection pool when exiting from "async with" block
        """
        await self.close()

    # initialize the connection pool
    async def initialize(self) -> None:
        """
        creates the asyncpg connection pool and registers the
        pgvector codec so Python can read and write VECTOR columns
        this is where we actually connect to Cloud SQL
        """
        logger.info(
            f"Connecting to Cloud SQL! | "
            f"host = {settings.db_host} | "
            f"database = {settings.db_name}"
        )

        try:
            self._pool = await asyncpg.create_pool(
                host = settings.db_host,
                port = int(settings.db_port),
                database = settings.db_name,
                user = settings.db_user,
                password = settings.db_password,
                min_size = POOL_MIN_SIZE,
                max_size = POOL_MAX_SIZE,
                init = self._init_connection,
                # init = means : "run this function on every new connection the pool creates" 
                # we use it to register our pgvector codec
            )
            
        except Exception:
            logger.exception("Failed to create Cloud SQL connection pool!")
            raise

        logger.info("Connection pool created successfully!")

    async def _init_connection(self, conn : asyncpg.Connection) -> None:
        """
        runs on every new database connection

        this is where we register the pgvector codec
        the translator that teaches asyncpg how to convert
        between Python lists and PostgreSQL vector columns

        wiithout codec:
            saving a chunk -> TypeError: cannot convert list to VECTOR
            reading a chunk -> asyncpg.exceptions.UndefinedTypeError

        with codec:
            Python [0.023, -0.041, 0.891, ...] <=> PostgreSQL VECTOR(1536)
            the conversion happens automatically

        args:
            conn: a new asyncpg connection, just created by the pool
        """

        await conn.set_type_codec(
            "vector",
            encoder = lambda v: json.dumps(v),
            # encoder: Python → PostgreSQL
            # when we save a chunk, asyncpg needs to convert our
            # python list [0.023, -0.041, ...] into something
            # PostgreSQL understands
            # json.dumps([0.023, -0.041]) → "[0.023, -0.041]"
            # postgreSQL's pgvector accepts this JSON string format

            decoder = lambda v: json.loads(v),
            # decoder: PostgreSQL → Python
            # when we read a chunk back, pgvector returns the
            # vector as a string "[0.023, -0.041, ...]"
            # json.loads converts it back to a Python list

            schema="public",
            format="text"
        )

    # close the connection pool
    async def close(self) -> None:
        """
        gracefully closes all database connections in the pool
        always call this when done with the VectorStore
        leaving connections open wastes Cloud SQL resources
        """

        if self._pool:
            await self._pool.close()
            logger.info("Connection pool closed")

    # save embedded chunks to Cloud SQL
    async def save_embedded_chunks(
        self,
        chunks: list[EmbeddedChunk]
    ) -> int:
        """
        saves a list of EmbeddedChunks into the chunks table

        uses INSERT ... ON CONFLICT DO NOTHING so it is safe
        to run multiple times - duplicate chunks are silently
        skipped instead of causing an error

        args:
            chunks: list of EmbeddedChunk objects to save

        returns:
            number of chunks successfully saved
        """

        if not chunks:
            logger.warning("save_embedded_chunks called with empty list!")
            return 0

        saved_count = 0

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            # acquire() checks out one connection from the pool
            # when this block exits, the connection is returned
            # to the pool automatically - not closed, just returned
            # this is efficient - the next save call reuses it

            for chunk in chunks:
                try:
                    await conn.execute(
                        """
                        INSERT INTO chunks
                            (nct_id, chunk_text, embedding, chunk_index, source)
                        VALUES
                            ($1, $2, $3, $4, $5)
                        ON CONFLICT DO NOTHING
                        """,
                        # $1, $2, $3, $4, $5 are parameter placeholders
                        # asyncpg fills them in from the arguments below
                        # this is called a parameterised query
                        
                        # ON CONFLICT DO NOTHING means:
                        # if a chunk with this chunk_id already exists,
                        # skip it silently instead of raising an error

                        chunk.unique_id,     # $1
                        chunk.chunk_text,    # $2
                        chunk.embedding,     # $3 - pgvector codec converts this list
                        # to vector automatically
                        chunk.chunk_index,   # $4
                        chunk.source,        # $5
                    )

                    saved_count += 1

                except Exception as e:
                    logger.error(
                        f"Failed to save chunk | "
                        f"chunk_id = {chunk.chunk_id} | "
                        f"error = {e}"
                    )

        logger.info(
            f"Chunks saved! | "
            f"saved = {saved_count} | "
            f"total_input = {len(chunks)} | "
            f"skipped = {len(chunks) - saved_count}"
        )

        return saved_count

    # save one study in studies table
    async def save_study(
        self,
        study_data: dict[str, Any]
        # a dictionary containing all the study fields to save
        # comes from ParsedStudy.model_dump() - converting
        # the Pydantic object into a plain Python dict
    ) -> bool:
        """
        saves one study record into the studies table

        uses INSERT ... ON CONFLICT (nct_id) DO UPDATE so that
        if a study already exists, its fields get refreshed
        with the latest data instead of being skipped.

        args:
            study_data: Dictionary of study fields to save.
        """

        assert self._pool is not None

        try:

            async with self._pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO studies
                        (nct_id, title, sponsor, phase, status,
                        conditions, interventions, primary_outcome,
                        secondary_outcomes, start_date, completion_date,
                        results_posted, enrollment, gcs_path)
                    VALUES
                        ($1, $2, $3, $4, $5,
                        $6, $7, $8,
                        $9, $10, $11,
                        $12, $13, $14)
                    ON CONFLICT (nct_id) DO UPDATE SET
                        title            = EXCLUDED.title,
                        sponsor          = EXCLUDED.sponsor,
                        phase            = EXCLUDED.phase,
                        status           = EXCLUDED.status,
                        conditions       = EXCLUDED.conditions,
                        interventions    = EXCLUDED.interventions,
                        primary_outcome  = EXCLUDED.primary_outcome,
                        secondary_outcomes = EXCLUDED.secondary_outcomes,
                        start_date       = EXCLUDED.start_date,
                        completion_date  = EXCLUDED.completion_date,
                        results_posted   = EXCLUDED.results_posted,
                        enrollment       = EXCLUDED.enrollment,
                        gcs_path         = EXCLUDED.gcs_path
                    """,
                    # ON CONFLICT (nct_id) DO UPDATE SET means:
                    # if this nct_id already exists in the table,
                    # UPDATE all its fields with the new values
                    # EXCLUDED.field refers to the value we tried to INSERT
                    # this pattern is called "upsert"
                    # UPDATE if exists, INSERT if new

                    study_data.get("nct_id"),              # $1
                    study_data.get("title"),               # $2
                    study_data.get("sponsor"),             # $3
                    study_data.get("phase"),               # $4
                    study_data.get("status"),              # $5
                    study_data.get("conditions", []),      # $6
                    study_data.get("interventions", []),   # $7
                    study_data.get("primary_outcome"),     # $8
                    study_data.get("secondary_outcomes", []),  # $9
                    study_data.get("start_date"),          # $10
                    study_data.get("completion_date"),     # $11
                    study_data.get("results_posted"),      # $12
                    study_data.get("enrollment"),          # $13
                    study_data.get("gcs_path"),            # $14
                )

                return True

        except Exception:
            logger.exception(
                f"Failed to save study {study_data.get('nct_id')}"
            )
            return False


    # semantic search
    async def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_DEFAULT,
        source_filter: str | None = None,
        # optional filter -> "study", "paper", or None (both)
        # lets agents search only study chunks or only paper chunks
        nct_id_filter: str | None = None
        # optional filter -> search only chunks from a specific study
        # useful when an agent is analysing one specific trial

    ) -> list[dict[str, Any]]:
        """
        finds the most semantically similar chunks to a query embedding
        uses pgvector's cosine distance operator (<=>) to compare
        the query embedding against every stored chunk embedding
        and returns the TOP_K closest ones

        this is the method every agent calls when it needs context
        It is the bridge between a natural language question and
        the relevant chunks stored in Cloud SQL

        returns:
            list of dictionaries, each containing:
            - nct_id:      which study this chunk belongs to
            - chunk_text:  the actual text content
            - chunk_index: position in the original document
            - source:      "study" or "paper"
            - distance:    cosine distance (lower = more similar)
                           0.0 = identical meaning
                           1.0 = completely different meaning
                           2.0 = opposite meaning
        """
        # we add WHERE clauses only if filters were provided
        conditions = []
        # list of SQL WHERE conditions to add if filters are set
        params: list[Any] = [query_embedding]
        # starts with the query embedding as $1
        # additional parameters ($2, $3) are added as filters are added
        param_count = 1
        # tracks our parameter numbering ($1, $2, $3...)
        # we increment this each time we add a filter parameter

        if source_filter:
            param_count += 1
            conditions.append(f"source = ${param_count}")
            params.append(source_filter)
            # example: source = $2 with params = [embedding, "study"]

        if nct_id_filter:
            param_count += 1
            conditions.append(f"nct_id = ${param_count}")
            params.append(nct_id_filter)
            # example: nct_id = $3 with params = [embedding, "study", "NCT123"]

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)
            # joins all conditions with AND:
            # "WHERE source = $2 AND nct_id = $3"

        param_count += 1
        params.append(top_k)

        query = f"""
            SELECT
                nct_id,
                chunk_text,
                chunk_index,
                source,
                embedding <=> $1 AS distance
            FROM chunks
            {where_clause}
            ORDER BY distance ASC
            LIMIT ${param_count}
        """
        # embedding <=> $1
        # the <=> operator is pgvector's cosine distance
        # it compares each stored embeddings against our query embedding
        # returns a number between 0 and 2
        # 0 = identical, 1 = orthogonal (unrelated), 2 = opposite

        # ORDER BY distance ASC
        # sort by distance, smallest first
        # smallest distance = most similar meaning = most relevant

        # LIMIT $N
        # return only the top N results

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)
            # conn.fetch() runs the query and returns ALL matching rows
            # *params unpacks our list into separate arguments:
            # [embedding, "study", 5] → $1=embedding, $2="study", $3=5

        results = [dict(row) for row in rows]
        # convert each asyncpg Record object into a plain Python dict
        # asyncpg returns its own Record type - dicts are easier
        # for the rest of our code to work with

        logger.info(
            f"Semantic search completed! | "
            f"results_found = {len(results)} | "
            f"top_k = {top_k} | "
            f"source_filter = {source_filter} | "
            f"nct_id_filter = {nct_id_filter}"
        )

        return results

    # how many chunks are stored
    async def get_chunk_count(self) -> int:
        """
        returns the total number of chunks currently in the database
        used by run_processing.py to report progress after saving

        returns:
            total count of rows in the chunks table
        """

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM chunks")
            # fetchval() returns a single value

        logger.info(f"Total chunks in database: {result}")
        return result


    # get all chunks from a study
    async def get_all_chunks_for_study(
        self,
        nct_id: str
    ) -> list[dict[str, Any]]:
        """
        returns every chunk belonging to a study, ordered by chunk_index

        args:
            nct_id: ClinicalTrials.gov study identifier

        returns:
            a list of dictionaries containing the study's chunks
        """

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    nct_id,
                    chunk_text,
                    chunk_index,
                    source
                FROM chunks
                WHERE nct_id = $1
                ORDER BY chunk_index ASC
                """,
                nct_id,
            )

        results = [dict(row) for row in rows]

        logger.info(
            f"Retrieved study chunks | "
            f"nct_id = {nct_id} | "
            f"chunks = {len(results)}"
        )

        return results


    # check if a study has already been processed
    async def study_exists(self, nct_id: str) -> bool:
        """
        checks if a study already has chunks saved in the database

        used by run_processing.py to skip studies that were already
        processed in a previous run

        returns:
            true if this study already has chunks in the database
            false if it has not been processed yet
        """

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE nct_id = $1",
                nct_id
            )

        return count > 0