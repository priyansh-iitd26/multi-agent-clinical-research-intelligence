# this file gives the system the ability to remember / recall what it found
# in past analysis sessions - and search through those memories semantically

# episodic memory analogue ->
# think of a detective who keeps a case notebook
# every time they investigate a case, they write down what they
# found, who was involved, and what conclusions they drew
# next time a similar case comes up, they flip through the notebook
# and ask "have I seen something like this before ?"
# that is exactly what episodic memory does for our agents
# every time an agent runs and finds a signal, that session
# gets saved as an episode - a record of what was investigated,
# what was found, and what the agent concluded

# one episode => one agent reasoning session
# example episode:
#     agent_name         : "missing_results_agent"
#     nct_id             : "NCT04788680"
#     episode_content    : "Investigated Novo Nordisk trial. Completed May 2019.
#                           Results never posted. 1200 participants enrolled."
#     outcome            : "signal_generated"

# this gets saved with a vector embedding of the episode_content so that
# future searches can find it semantically

# so, every agent run builds on it's previous runs
# before investigating, the agent asks: "what do I already know ?"
# it searches its past episodes / experiences and uses that context
# the system gets smarter every time it runs

import json
import uuid
# uuid stands for "Universally Unique Identifier"
# uuid.uuid4() generates a random ID like:
# "a3f8c2d1-4e5b-6f7a-8b9c-0d1e2f3a4b5c"
# it is "universally unique" - the probability of two UUIDs
# being the same is astronomically small (essentially impossible)
# we use uuid here to give every episode its own unique ID so no
# two episodes ever clash in the database

from datetime import datetime, UTC
from openai import AsyncOpenAI
import asyncpg
from config.settings import settings

from config.logging_config import setup_logging

logger = setup_logging(__name__)

# EpisodicStore Class
# EpisodicStore is a blueprint for an object that knows how to:
#   - connect to our Cloud SQL database
#   - save agent reasoning sessions as episodes
#   - search through past episodes by meaning
#   - return recent episodes

class EpisodicStore:
    """
    stores and retrieves agent reasoning sessions as episodes

    each episode is one agent's reasoning session - what it
    investigated, what it found, and what it concluded

    episodes are embedded and stored so future sessions can
    search through past findings by meaning

    usage:
        store = EpisodicStore()

        # saving what an agent found
        await store.save_episode(
            agent_name="missing_results_agent",
            nct_id="NCT04788680",
            content="Novo Nordisk trial completed 2019. Results never posted.",
            outcome="signal_generated"
        )

        # search past episodes by meaning
        past = await store.search_episodes(
            query="sponsor never posted results",
            agent_name="missing_results_agent",
            top_k=3
        )
    """

    def __init__(self):

        self._pool: asyncpg.Pool | None = None

        # for converting episode_content into 1536-number vector embeddings
        self._openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self._embedding_model = settings.openai_embedding_model
        # NOTE: the embedding model must match the embedding model used in embedder.py
        # because episode embeddings and chunk embeddings must be
        # in the same vector space to be comparable

        logger.info("EpisodicStore initialised!")

    # private helper method : _ensure_pool
    #
    # what is a private method ?
    # methods starting with underscore (_) are private by convention
    # they are internal helpers - only called by other methods
    # inside this class, never directly from outside
    # example: store._ensure_pool()  ← we will never write this,
    #           instead it is called automatically by save_episode()
    #          and search_episodes() whenever they need the database connection

    async def _ensure_pool(self) -> None:
        """
        makes sure the database connection pool is open

        based on concept of lazy initialization
        instead of opening the database connection the moment
        EpisodicStore() is created, we wait until someone actually
        tries to use it ("lazy initialization")

        when the entire system starts, it imports many classes including
        EpisodicStore 
        if we opened the database connection in __init__, 
        every import would immediately try to connect
        to Cloud SQL - even if that class is never actually used
        in that run
        lazy initialization avoids wasted connections
        """

        if self._pool is not None:
            # self._pool is not None means the pool already exists
            # nothing to do - return immediately
            return

        # self._pool is None here

        self._pool = await asyncpg.create_pool(
            # create_pool() creates a group of database connections
            # that stay open and ready to use
            # we "await" it because connecting to the database is a
            # network operation that takes time - await lets other
            # async tasks run while we wait

            host = settings.db_host,
            port = settings.db_port,
            database = settings.db_name,
            user = settings.db_user,
            password = settings.db_password,
            min_size = 1,
            # keeping at least 1 connection open at all times
            max_size=5,
            # allow at most 5 simultaneous connections

            # EpisodicStore is called less frequently than VectorStore
            # (agents check memory at the START of a session, not
            # on every query), so 5 connections is more than enough
            # Using fewer connections also reduces Cloud SQL load

            init = self._init_connection
        )

        logger.info("EpisodicStore pool created!")

    # @staticmethod is a method that belongs to the CLASS
    # but does NOT have access to the instance (self) or the
    # class itself (cls)

    # _init_connection does not need to access ANY instance
    # variables - it only works with the "conn" parameter passed
    # to it by asyncpg
    # using @staticmethod makes this explicit
    # and prevents accidental use of self inside the method
    # also slightly more memory-efficient

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """
        registers the custom pgvector VECTOR codec on every new connection

        asyncpg calls this function automatically on every new
        connection it creates, via the init = parameter in _ensure_pool()
        """

        await conn.set_type_codec(
            "vector",
            encoder = lambda v: json.dumps(v),
            decoder = lambda v: json.loads(v),
            schema = "public"
        )

    # private helper method : _embed (embedding episodic content)

    async def _embed(self, text: str) -> list[float]:
        """
        converts any text string (episodic_content) into a 1536-number vector embedding

        we use this method in TWO places:
        1. when SAVING an episode - embed the content so it can
           be found by semantic search later
        2. when SEARCHING episodes - embed the query so we can
           compare it against stored episode embeddings
        """

        response = await self._openai.embeddings.create(
            # embeddings.create() sends text to OpenAI and gets
            # back vector embeddings 
            # we await it because it is a network request

            model = self._embedding_model,
            input = text,
            # here we pass one string at a time because episodes
            # are saved and searched one at a time, unlike the
            # processing layer where we batched 50 chunks at once in an API call
        )

        return response.data[0].embedding

    # public core method : save_episode

    async def save_episode(
        self,
        agent_name: str,
        episodic_content: str,
        # what the agent investigated and what it found
        # this is the main body of the episode
        # example:
        # "Investigated NCT04788680. Sponsor: Novo Nordisk.
        #  Trial completed May 2019. Results never posted.
        #  Enrolled 1200 participants. Phase 3 diabetes study."
        # this is what gets converted to 1536 numbers and stored
        # future searches compare against this content

        nct_id: str | None = None,
        outcome: str | None = None,
        # what happened as a result of this investigation
        # example values:
        # "signal_generated"  -> agent found something worth flagging
        # "no_signal"         -> agent investigated but found nothing
        # "sent_to_review"    -> signal was low confidence, sent to HITL
        # used for filtering and analytics
    ) -> str:
        """
        saves one agent reasoning session as an episode in Cloud SQL episodes table

        1. makes sure the database connection is open
        2. generates a unique ID for this episode
        3. converts the content to a 1536-number embedding
        4. inserts all of this into the episodes table
        5. returns the episode_id
        """

        # step-1 : making sure the database connection pool is open
        # if the pool is already open, this returns immediately
        # if not, it creates the pool first
        await self._ensure_pool()

        # step-2 : generating unique id for every episode
        episode_id = str(uuid.uuid4())
        # uuid.uuid4() generates a random unique ID object
        # str() converts it to a plain string we can store in the db
        # example result: "a3f8c2d1-4e5b-6f7a-8b9c-0d1e2f3a4b5c"

        # step-3 : converting the episodic content into an embedding
        embedding = await self._embed(episodic_content)

        # step-4 : inserting in episodes table of our Cloud SQL instance - "clinical_trials_db"

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            # self._pool.acquire() checks out one connection from the pool
    
            await conn.execute(
                # conn.execute() runs a SQL statement on the database
                # we await it because sending SQL to the database is
                # a network operation that takes time

                # execute() is used for INSERT, UPDATE, DELETE
                # statements that do not return rows
                # (for SELECT statements we use fetch() instead)

                """
                INSERT INTO episodes (
                    episode_id,
                    agent_name,
                    nct_id,
                    content,
                    outcome,
                    embedding,
                    created_at
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                """,

                episode_id,       # $1
                agent_name,       # $2
                nct_id,           # $3
                episodic_content, # $4
                outcome,          # $5
                embedding,        # $6
                datetime.now(UTC) # $7
            )

        logger.info(
            f"Episode saved! | "
            f"agent = {agent_name} | "
            f"nct_id = {nct_id} | "
            f"outcome = {outcome} | "
            f"episode_id = {episode_id}"
        )

        return episode_id

    # public core method : search_episodes

    async def search_episodes(
        self,
        query: str,
        # the question we are asking about past sessions
        # this is converted to a 1536-number embedding and compared
        # against all stored episode embeddings
        # example: "find episodes where sponsor never posted results"
        # example: "find episodes about Novo Nordisk"
        # the search is semantic
        agent_name: str | None = None,
        top_k: int = 5,
        min_similarity: float = 0.5,
        # minimum similarity score for a result to be included in top_k
        # 1.0 => identical meaning
        # 0.5 => at least 50% similar in meaning
        # 0.0 => completely different
        # episodes below 0.5 similarity are probably not relevant
        # and would just add noise to the agent's context
    ) -> list[dict]:
        """
        searches past episodes by semantic similarity to a query

        1. makes sure the database connection is open
        2. converts the query text to a 1536-number embedding
        3. builds a SQL query with optional filters
        4. uses pgvector's <=> operator to find similar episodes
        5. returns the top_k most similar episodes as dictionaries
        """

        # step-1
        await self._ensure_pool()

        # step-2
        query_embedding = await self._embed(query)

        # step-3 : building SQL query dynamically
        sql = """
            SELECT
                episode_id,
                agent_name,
                nct_id,
                content,
                outcome,
                created_at,
                1 - (embedding <=> $1) AS cosine_similarity
            FROM episodes
            WHERE 1 - (embedding <=> $1) >= $2

            -- $1 = query_embedding, $2 = min_similarity
        """
        params: list = [query_embedding, min_similarity] # $1, $2, ...

        param_idx = 3
        # param_idx tracks the next parameter number to use
        # we start at 3 because $1 and $2 are already used

        if agent_name:
            # only add this filter if agent_name was provided
            # if agent_name is None (the default), we skip this block
            # and search all agents' episodes
            sql += f" AND agent_name = ${param_idx}"
            params.append(agent_name)
            param_idx += 1

        sql += f"""
            ORDER BY embedding <=> $1
            LIMIT ${param_idx}
        """
        params.append(top_k)

        # executing the search SQL query in db

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            rows = await conn.fetch(sql, *params)
            # conn.fetch() runs a SELECT query and returns all rows
        
        episodes = [
            {
                "episode_id": row["episode_id"],
                "agent_name": row["agent_name"],
                "nct_id":     row["nct_id"],
                "content":    row["content"],
                "outcome":    row["outcome"],
                "similarity": round(float(row["similarity"]), 3),
                "created_at": str(row["created_at"])
            }
            for row in rows
        ]

        logger.info(
            f"Episode search completed! | "
            f"query = '{query[:50]}...' | "
            # preventing extremely long queries from flooding the log
            f"agent_filter = {agent_name} | "
            f"results_found = {len(episodes)}"
        )

        return episodes

    # utility method : get_recent_episodes

    async def get_recent_episodes(
        self,
        agent_name: str | None = None,
        limit: int = 10
    ) -> list[dict]:
        """
        returns the most recent episodes, newest first

        ** unlike search_episodes() which finds episodes by meaning,
        this method finds episodes by timestamp - just the most recent ones

        used by the API endpoint GET /api/v1/memory/episodes so
        analysts can browse what the agents have been doing recently
        """

        await self._ensure_pool()

        if agent_name:
            sql = """
                SELECT episode_id, agent_name, nct_id,
                       content, outcome, created_at
                FROM episodes
                WHERE agent_name = $1
                ORDER BY created_at DESC
                LIMIT $2
            """
            params = [agent_name, limit]
    
        else: # agent_name not provided -> don't filter by agent_name
            sql = """
                SELECT episode_id, agent_name, nct_id,
                       content, outcome, created_at
                FROM episodes
                ORDER BY created_at DESC
                LIMIT $1
            """
            params = [limit]


        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        return [
            {
                "episode_id": row["episode_id"],
                "agent_name": row["agent_name"],
                "nct_id":     row["nct_id"],
                "content":    row["content"],
                "outcome":    row["outcome"],
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

    # utility method : count_episodes

    async def count_episodes(
        self,
        agent_name: str | None = None,
    ) -> int:
        """
        returns the total number of episodes stored.

        used by the health check endpoint to show how much memory
        the system has accumulated - how many past sessions exist
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            if agent_name:
                count = await conn.fetchval(
                    # fetchval() is like fetch() but returns a SINGLE VALUE
                    # instead of a list of rows
                    # fetchval() is perfect for COUNT(*) queries that return one number
                    # fetch() would return [{"count": 47}] - a list with one dict
                    # fetchval() returns just 47 - the number directly
                    "SELECT COUNT(*) FROM episodes WHERE agent_name = $1",
                    agent_name,
                )

            else: # agent_name not provided
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM episodes"
                )

        return count or 0

    # close the connection pool and release all connections

    async def close(self) -> None:
        """
        we call this when the application shuts down cleanly
        without closing, connections may stay open on Cloud SQL
        unnecessarily - wasting resources and potentially hitting
        connection limits

        in FastAPI, this is called in the lifespan shutdown handler
        """

        if self._pool:
            
            await self._pool.close()

            self._pool = None

            logger.info("EpisodicStore pool closed!")