# this file takes every TextChunk produced by chunker.py and
# converts it into a vector embedding -> a list of 1536 numbers
# that mathematically capture the meaning of that chunk

# when our Missing Results agent asks "find studies where sponsor
# never posted results", we convert that query into 1536 numbers
# then pgvector finds the chunks whose numbers are closest
# to the query's numbers -> semantic search

# why text-embedding-3-small AND NOT text-embedding-3-large ?
# text-embedding-3-large produces 3072 dimensions but pgvector's
# hnsw index has a 2000-dimension hard limit
# we discovered this during the build — the index creation failed 
# text-embedding-3-small produces 1536 dimensions — well within
# the limit, cheaper, faster, and more than sufficient quality
# for clinical trial signal detection

# batching -> why we don't embed one chunk at a time:
# if we sent one API call to OpenAIEmbeddings per chunk, 300 chunks = 300 API calls
# which is very slow and expensive
# OpenAI accepts up to 100 chunks in a single API call
# we batch 50 at a time for fast, efficient and within OpenAI's rate limits comfortably

# OpenAI calls are network requests and we want our program to stay responsive while waiting
import asyncio

from dataclasses import dataclass
from openai import AsyncOpenAI
# AsyncOpenAI is OpenAI's official async Python client
# the "Async" version lets us use await - meaning our program
# does not freeze while waiting for OpenAI to respond
# regular OpenAI() client is synchronous and would block everything

from processing.chunker import TextChunk
# we will receive TextChunk objects as INPUT and return
# EmbeddedChunk objects as OUTPUT 
# TextChunk is defined in chunker.py

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

# configuration

# how many chunks to send to OpenAIEmbedding in one API call
# OpenAI accepts up to 100 inputs per call
# we use 50 as a safe, comfortable batch size that:
#   - stays well within OpenAI's limits
#   - processes 300 chunks in just 6 API calls instead of 300
#   - gives us a natural retry unit if one batch fails
BATCH_SIZE = 50

# how many times to retry a failed OpenAIEmbedding API call
# OpenAI occasionally returns 429 (rate limit) or 500 (server error)
RETRY_ATTEMPTS = 3

# how long to wait between retry attempts
# 2 seconds gives OpenAI time to recover from a rate limit hit
RETRY_SLEEP_SECONDS = 2

# EmbeddedChunk dataclass
# **output format of the embedder
@dataclass
class EmbeddedChunk:
    """
    a TextChunk that has been enriched with its vector embedding

    this is what gets saved to -> "Cloud SQL chunks table"
    every field from TextChunk is carried over, plus one new field:
    embedding — the list of 1536 numbers representing this chunk's meaning

    fields:
        chunk_id:    unique identifier for chunk (example: "NCT04788680_chunk_0")
        unique_id:   which study/paper this chunk belongs to
        chunk_text:  the actual text content
        chunk_index: position in the original document (0, 1, 2...)
        source:      "study" or "paper"
        word_count:  number of words in this chunk
        embedding:   1536 floating point numbers from OpenAIEmbedding
    """

    chunk_id : str
    unique_id : str
    chunk_text : str
    chunk_index : int
    source : str
    word_count : int
    embedding: list[float]

# embedder class
class Embedder:
    """
    converts TextChunks into EmbeddedChunks using OpenAI's text-embedding-3-small model

    processes chunks in batches of size = BATCH_SIZE for efficiency
    automatically retries failed API calls up to RETRY_ATTEMPTS times
    """

    def __init__(self):
        # creating the async OpenAI client using our API key from .env
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        # this client is reused for every API call in this session

        self._model = settings.openai_embedding_model
        # stored as an instance variable so it is easy to see
        # and change in one place if needed

        logger.info(
            f"Embedder initialised! | model={self._model}"
        )

    # embed a list of chunks
    async def embed_chunks(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        """
        converts a list of TextChunks into EmbeddedChunks

        splits the input into batches of BATCH_SIZE and processes
        each batch with one OpenAI API call

        args:
            chunks: List of TextChunk objects from chunker.py

        returns:
            a list of EmbeddedChunk objects — same chunks but now
            each has a 1536-number vector embedding attached
            any chunk that fails to embed is skipped (not fatal)
        """

        # if nothing to embed, returning empty list
        if not chunks:
            logger.warning("embed_chunks called with empty list -> nothing to do")
            return []

        logger.info(
            f"Starting embedding | "
            f"total_chunks={len(chunks)} | "
            f"batch_size={BATCH_SIZE} | "
            f"embedding_model={self._model}"
        )

        all_embedded: list[EmbeddedChunk] = []

        # split the full list into smaller batches
        # (example: 150 chunks -> 3 batches of size 50 each)
        batches = self._create_batches(chunks)

        for batch_num, batch in enumerate(batches):
            logger.info(
                f"Embedding batch {batch_num + 1} / {len(batches)} | "
                f"chunks_in_batch = {len(batch)}"
            )

            # embed this batch with automatic retry on failure
            embedded_batch = await self._embed_batch_with_retry(
                batch = batch,
                batch_num = batch_num,
            )

            all_embedded.extend(embedded_batch)

            if batch_num < len(batches) - 1:
                await asyncio.sleep(0.5)

        logger.info(
            f"Embedding completed! | "
            f"total_embedded = {len(all_embedded)} | "
            f"total_input = {len(chunks)} | "
            f"skipped = {len(chunks) - len(all_embedded)}"
        )

        return all_embedded

    # private method : _create_batches
    def _create_batches(self, chunks: list[TextChunk]) -> list[list[TextChunk]]:
        """
        splits a list of TextChunk chunks into smaller batches
        example:
            150 chunks with BATCH_SIZE = 50 =>
            [[chunk_0..chunk_49], [chunk_50..chunk_99], [chunk_100..chunk_149]]

        args:
            chunks: the full list of chunks to split

        returns:
            a list of lists - each inner list is one batch
        """

        return [
            chunks[i : i + BATCH_SIZE]
            for i in range(0, len(chunks), BATCH_SIZE)
        ]

    # private method : _embed_batch_with_retry
    async def _embed_batch_with_retry(
            self, 
            batch: list[TextChunk], 
            batch_num: int
    ) -> list[EmbeddedChunk]:
        """
        embeds one batch of chunks, retrying on failure

        attempts the embedding up to RETRY_ATTEMPTS times
        waits RETRY_SLEEP_SECONDS between each attempt
        
        returns an empty list if all attempts fail — the pipeline
        continues with the remaining batches rather than crashing

        args:
            batch : one batch of TextChunks to embed
            batch_num : the batch number (for logging purpose only)

        returns:
            a list of EmbeddedChunks for this batch
            empty list if all retry attempts failed
        """

        for attempt in range(1, RETRY_ATTEMPTS + 1):

            try:
                return await self._embed_batch(batch = batch)

            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    # we still have retries left - logging a warning and waiting
                    logger.warning(
                        f"Embedding failed | "
                        f"batch = {batch_num + 1} | "
                        f"attempt_number = {attempt} / {RETRY_ATTEMPTS} | "
                        f"error = {e} | "
                        f"retrying in {RETRY_SLEEP_SECONDS}sec..."
                    )

                    # wait before retrying
                    # asyncio.sleep() is non-blocking — other async
                    # tasks can run during this pause
                    await asyncio.sleep(RETRY_SLEEP_SECONDS)

                else:
                    # this was the last attempt and it still failed to embed
                    # logging an error and giving up on this batch
                    logger.error(
                        f"Embedding failed after {RETRY_ATTEMPTS} attempts! | "
                        f"batch = {batch_num + 1} | "
                        f"error = {e} | "
                        f"skipping this batch..."
                    )

                    return []
                    # losing one batch of chunks is not fatal

        return [] # just a safety fallback (never executed)

    # private method : _embed_batch

    async def _embed_batch(self, batch: list[TextChunk]) -> list[EmbeddedChunk]:
        """
        makes one OpenAI API call to embed an entire batch of chunks

        this is the method that actually talks to OpenAI
        it sends up to BATCH_SIZE chunk texts in one request 
        and gets back one embedding per chunk

        args:
            batch: one batch of TextChunks (up to BATCH_SIZE)

        returns:
            list of EmbeddedChunks with embeddings attached

        raises:
            exception: if the OpenAI API call fails
                       ** the caller (_embed_batch_with_retry) handles this
        """

        # extracting just the text (to embed) from each chunk
        # we send: ["Study NCT04788680 TITLE: ...", "SPONSOR: ...", ...]
        texts = [chunk.chunk_text for chunk in batch]

        response = await self._client.embeddings.create(
            model = self._model,
            # value: "text-embedding-3-small"
            # this produces 1536-dimensional embeddings

            input = texts
            # the list of texts to embed
            # OpenAI processes ALL of them in one shot (one API call)
            # and returns one embedding per text in the same order
        )

        # await means: pause here until OpenAI responds
        # the response contains one embedding object per input text
        # response.data is a list of embedding objects
        # response.data[0].embedding is the first list of 1536 numbers
        # response.data[1].embedding is the second list of 1536 numbers
        # ...

        embedded_chunks: list[EmbeddedChunk] = []

        for i, chunk in enumerate(batch):
            # OpenAI GUARANTEES the response order matches input order

            # get the embedding for THIS specific chunk
            # response.data[i] is the ith result from OpenAI call
            # .embedding is the list of 1536 floats
            embedding_vector = response.data[i].embedding

            embedded_chunk = EmbeddedChunk(
                chunk_id = chunk.chunk_id,
                unique_id = chunk.unique_id,
                chunk_text = chunk.chunk_text,
                chunk_index = chunk.chunk_index,
                source = chunk.source,
                word_count = chunk.word_count,
                embedding = embedding_vector,
                # all fields are copied from the original TextChunk
                # and the embedding is added as the new field
            )

            embedded_chunks.append(embedded_chunk)

        logger.info(
            f"Batch embedded successfully! | "
            f"chunks = {len(embedded_chunks)} | "
            f"embedding_dims = {len(embedded_chunks[0].embedding) if embedded_chunks else 0}"
        )
        # logging the embedding dimensions as a sanity check
        # it should always show 1536 for text-embedding-3-small
        # if it ever shows something different => something is wrong

        return embedded_chunks