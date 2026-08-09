# this file manages the SPONSOR KNOWLEDGE BASE - a growing collection
# of FACTS about every research sponsor our system has ever encountered

# semantic memory analogue ->
# think of a credit rating agency like CIBIL in India
# every time you take a loan or miss a payment,
# your credit score changes - the agency builds up a profile of your
# financial behaviour over years of transactions

# that is exactly what semantic memory does for sponsors
# every time our system analyzes a study, it updates the sponsor's profile:
#     - did they post results on time? -> credibility goes up
#     - did they miss results? -> credibility goes down
#     - did they switch outcomes? -> broken promises count goes up
#     - did they delay the trial silently? -> average delay goes up
# hence, over time, the system builds a rich, data-driven picture of every
# sponsor's behaviour - not based on reputation, but on evidence

# EPISODIC vs PROCEDURAL vs SEMANTIC
# Episodic => WHAT happened in past sessions (case diary)
#                "I found missing results for NCT04788680 on March 15"
# Procedural => HOW to reason - the rules (rulebook)
#                "Do not flag terminated trials as missing results"
# Semantic => FACTS about the world - accumulated knowledge (encyclopedia)
#                "Novo Nordisk: 47 studies, credibility 0.82, 2 broken promises"

# where sponsor profiles are stored :
# in the existing "sponsor_profiles" table in Cloud SQL

import asyncpg
from datetime import datetime, UTC
from typing import Any

from config.settings import settings

from config.logging_config import setup_logging

logger = setup_logging(__name__)

class SemanticStore:
    """
    manages the sponsor knowledge base - credibility profiles
    built up over time as our system analyzes more studies

    each sponsor gets one profile row in the sponsor_profiles table
    that row is updated (never replaced) every time new information
    about that sponsor is discovered during an analysis run

    usage:
        store = SemanticStore()

        # get what we know about a sponsor
        profile = await store.get_sponsor_profile("Novo Nordisk")

        # update after analyzing a study
        await store.update_sponsor_knowledge(
            sponsor="Novo Nordisk",
            results_posted=True,
            had_broken_promise=False,
            delay_days=5
        )
    """

    def __init__(self):

        self._pool: asyncpg.Pool | None = None

        logger.info("SemanticStore initialized!")


    # private helper : _ensure_pool
    async def _ensure_pool(self) -> None:
        """
        creates the connection pool if it does not exist yet
        called at the start of every public method
        same pattern as EpisodicStore and ProceduralStore
        """

        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            host = settings.db_host,
            port = settings.db_port,
            database = settings.db_name,
            user = settings.db_user,
            password = settings.db_password,
            min_size = 1,
            max_size=5
        )

        logger.info("SemanticStore connection pool created!")


    # public core method : get_sponsor_profile
    async def get_sponsor_profile(
        self,
        sponsor: str,
        # the name of the sponsor to look up
        # example: "Novo Nordisk", "Pfizer", "National Cancer Institute"
        # must match exactly how the sponsor appears in the studies table
    ) -> dict[str, Any] | None:
        """
        retrieves everything we know about a specific sponsor

        returns:
        a dictionary containing the sponsor's full profile:
          - sponsor           -> the sponsor's name
          - credibility_score -> 0.0 (worst) to 1.0 (best)
          - total_studies     -> how many studies we have analysed
          - results_posted    -> how many times they posted results on time
          - results_missing   -> how many times results were NOT posted
          - broken_promises   -> how many outcome switches detected
          - avg_delay_days    -> average days late on timeline
          - last_updated      -> when this profile was last modified
        None if nothing is found for that sponsor

        ** calculation of credibility score:
            it weights different factors:
                70% -> results compliance rate (posted / total studies)
                30% -> promise keeping (reduced per broken promise)
        range: 0.0 to 1.0. below 0.6 triggers a LOW_CREDIBILITY signal
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            row = await conn.fetchrow(
                # fetchrow() returns ONE row - not a list of rows
                # this is correct here because each sponsor has exactly
                # one profile row in the sponsor_profiles table
                # if no row exists for this sponsor, fetchrow() returns None

                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                WHERE sponsor = $1
                """,
                sponsor
            )

        if row is None:
            # fetchrow() returns None when no matching row exists
            # this means we have never analysed a study from this sponsor
            # return None - the caller decides what to do with a new sponsor
            # the Track Record agent handles this: "first time seeing this
            # sponsor - not enough data to make a reliable judgment."
            logger.info(
                f"No profile found for sponsor | sponsor={sponsor}"
            )
            return None

        return {
            "sponsor": row["sponsor"],
            "credibility_score": float(row["credibility_score"] or 0.0),
            # float() converts the database decimal type to python float
            # "or 0.0" handles the case where the value is NULL in the DB -
            # NULL or 0.0 evaluates to 0.0, giving us a safe default
            "total_studies": int(row["total_studies"] or 0),
            "results_posted": int(row["results_posted"] or 0),
            "results_missing": int(row["results_missing"] or 0),
            "broken_promises": int(row["broken_promises"] or 0),
            "avg_delay_days": float(row["avg_delay_days"] or 0.0),
            "last_updated": str(row["last_updated"])
        }


    # public core method : update_sponsor_knowledge
    async def update_sponsor_knowledge(
        self,
        sponsor: str,
        # the sponsor name we want to update
        # if this sponsor does not exist yet - we CREATE a new profile (row)
        # if they already exist - we UPDATE the existing profile
        # this create-or-update pattern is called "upsert" in databases

        results_posted: bool = False,
        # did this sponsor post results for the study we just analyzed?
        # true  -> compliance count goes up, credibility improves
        # false -> missing count goes up, credibility decreases
        
        had_broken_promise: bool = False,
        # did we detect outcome switching in the study we just analyzed?
        # True  -> broken promises count goes up, credibility decreases
        # False -> no change to broken promises count

        delay_days: int = 0,
        # how many days past the completion date was this study?
        # 0 = on time or completed within the grace period
        # positive number = days late 
        # example: 45 means 45 days late
        # this updates the rolling average delay for this sponsor
    ) -> None:
        """
        updates a sponsor's profile with new information from one study
        
        uses the upsert pattern
        
        this is more efficient and safer than checking first with
        SELECT, then deciding whether to INSERT or UPDATE
        the two-step approach has a race condition - two agents
        running in parallel could both check, both see "not exists",
        and both try to INSERT, causing a duplicate key error
        UPSERT handles this atomically - it is thread-safe

        re-calculation of credibility score :
        after updating the counts, we recalculate credibility:
          compliance_rate = results_posted_count / total_studies
          promise_penalty = broken_promises * 0.3
          credibility = (compliance_rate * 0.7) - promise_penalty
          credibility = max(0.0, min(1.0, credibility))
          (clamped between 0.0 and 1.0 — cannot go negative or above 1)
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            await conn.execute(
                """
                INSERT INTO sponsor_profiles (
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                )
                VALUES ($1, 0.5, 1, $2, $3, $4, $5, NOW())
                ON CONFLICT (sponsor) DO UPDATE SET
                    total_studies   = sponsor_profiles.total_studies + 1,
                    results_posted  = sponsor_profiles.results_posted + $2,
                    results_missing = sponsor_profiles.results_missing + $3,
                    broken_promises = sponsor_profiles.broken_promises + $4,
                    avg_delay_days  = (
                        (sponsor_profiles.avg_delay_days *
                         sponsor_profiles.total_studies) + $5
                    ) / (sponsor_profiles.total_studies + 1),
                    last_updated    = NOW()
                """,
                # INSERT section (new sponsor - first time we see them):
                # credibility_score starts at 0.5 - neutral, no data yet
                # total_studies starts at 1 — this is their first study.
                # $2 = results_posted as int (True=1, False=0)
                # $3 = results_missing as int (True=1, False=0)
                # $4 = broken_promises as int (True=1, False=0)
                # $5 = delay_days (integer)
                #
                # ON CONFLICT (sponsor) DO UPDATE section (existing sponsor):
                # total_studies   -> increment by 1 (one more study analysed)
                # results_posted  -> add 1 if True, add 0 if False
                # results_missing -> add 1 if True, add 0 if False
                # broken_promises -> add 1 if True, add 0 if False
                #
                # avg_delay_days calculation :
                # we compute a RUNNING AVERAGE -> updating the average
                # without storing every individual delay value
                # formula: new_avg = (old_avg * old_count + new_value) / (old_count + 1)
                # example: old_avg=10 days, old_count=4, new_value=20 days
                #   new_avg = (10 * 4 + 20) / (4 + 1)
                #           = (40 + 20) / 5
                #           = 60 / 5
                #           = 12 days

                # sponsor_profiles.column_name refers to the EXISTING value
                # in the row BEFORE the update - this is PostgreSQL syntax
                # for referencing the current row's values in an UPDATE

                sponsor,
                int(results_posted),
                int(not results_posted),
                int(had_broken_promise),
                float(delay_days)
            )

            # recalculating credibility score
            # we do this as a SECOND query after the update above
            # separate because the credibility formula needs
            # the UPDATED counts (after step 1), not the old ones

            await conn.execute(
                """
                UPDATE sponsor_profiles
                SET credibility_score = GREATEST(0.0, LEAST(1.0,
                    (
                        CASE
                            WHEN total_studies = 0 THEN 0.5
                            ELSE (results_posted::float / total_studies) * 0.7
                        END
                    ) - (broken_promises * 0.3)
                ))
                WHERE sponsor = $1
                """,
                # CASE WHEN total_studies = 0 THEN 0.5
                #   -> if we somehow have no studies, default to 0.5 (neutral)
                #   -> this prevents division by zero
                #
                # ELSE (results_posted::float / total_studies) * 0.7
                #   -> results_posted::float casts the integer to float
                #      so PostgreSQL does decimal division, not integer division
                #      example: 3::float / 4 = 0.75, not 0 (integer division)
                #   -> divide posted results by total studies => compliance rate
                #      example: 40 posted / 47 total = 0.851
                #   -> multiply by 0.7 = 70% weight on compliance
                #      example: 0.851 * 0.7 = 0.596
                #
                # - (broken_promises * 0.3) -> 30% weight on broken promises
                #   -> each broken promise reduces credibility by 0.3
                #   -> final: 0.596 - 0.3 = 0.296
                #
                # GREATEST(0.0, LEAST(1.0, ...))
                #   → LEAST(1.0, value) caps the score at 1.0 maximum
                #   → GREATEST(0.0, value) prevents the score going negative
                #   → together they clamp the result between 0.0 and 1.0
                #   → a sponsor cannot have credibility above 1.0 or below 0.0

                sponsor
            )

        logger.info(
            f"Sponsor knowledge updated! | "
            f"sponsor = {sponsor} | "
            f"results_posted = {results_posted} | "
            f"broken_promise = {had_broken_promise} | "
            f"delay_days = {delay_days}"
        )


    # utility method : get_low_credibility_sponsors
    async def get_low_credibility_sponsors(
        self,
        threshold: float = 0.6,
        # we return sponsors whose credibility score is BELOW this number
        # default is 0.6 - sponsors below this are considered concerning
        min_studies: int = 3
        # default minimum number of studies required before we flag a sponsor
        # a sponsor with only 1 study and 1 issue might just be unlucky
        # we require at least 3 studies before making a judgment -
        # ensures our credibility assessment has enough data to be meaningful
    ) -> list[dict]:
        """
        returns all sponsors whose credibility is below the input threshold

        used by:
            1. the Track Record agent -> to quickly identify problematic sponsors
            2. the Pattern Finder agent -> to check if a sponsor is a repeat offender
            3. the API endpoint GET /api/v1/sponsors -> for analyst dashboards

        returns:
            a list of sponsor profile dicts ordered by credibility ascending
            lowest credibility (worst) sponsors appear first
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                WHERE credibility_score < $1
                  AND total_studies >= $2
                ORDER BY credibility_score ASC
                """,
                threshold,
                min_studies
            )

        sponsors = [
            {
                "sponsor":           row["sponsor"],
                "credibility_score": float(row["credibility_score"] or 0.0),
                "total_studies":     int(row["total_studies"] or 0),
                "results_posted":    int(row["results_posted"] or 0),
                "results_missing":   int(row["results_missing"] or 0),
                "broken_promises":   int(row["broken_promises"] or 0),
                "avg_delay_days":    float(row["avg_delay_days"] or 0.0),
                "last_updated":      str(row["last_updated"]),
            }
            for row in rows
        ]

        logger.info(
            f"{len(sponsors)} low credibility sponsors found! | "
            f"threshold = {threshold} | "
            f"min_studies = {min_studies}"
        )

        return sponsors


    # utility method : get_all_sponsor_profiles
    async def get_all_sponsor_profiles(
        self,
        limit: int = 50,
        # maximum number of sponsor profiles to return
        # default 50 - enough for an API dashboard view
    ) -> list[dict]:
        """
        returns all sponsor profiles ordered by credibility

        Used by the API for analytics dashboards - showing analysts
        the full picture of every sponsor we have knowledge about
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                FROM sponsor_profiles
                ORDER BY credibility_score ASC
                LIMIT $1
                """,
                limit
            )

        return [
            {
                "sponsor":           row["sponsor"],
                "credibility_score": float(row["credibility_score"] or 0.0),
                "total_studies":     int(row["total_studies"] or 0),
                "results_posted":    int(row["results_posted"] or 0),
                "results_missing":   int(row["results_missing"] or 0),
                "broken_promises":   int(row["broken_promises"] or 0),
                "avg_delay_days":    float(row["avg_delay_days"] or 0.0),
                "last_updated":      str(row["last_updated"]),
            }
            for row in rows
        ]


    # utility method : sponsor_exists
    async def sponsor_exists(self, sponsor: str) -> bool:
        """
        checks if a sponsor profile already exists in the database

        used before creating a new profile - avoids duplicate entries
        also used by agents to decide whether to load a profile or
        note that "we have never seen this sponsor before"
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            count = await conn.fetchval(
                "SELECT COUNT(*) FROM sponsor_profiles WHERE sponsor = $1",
                sponsor
            )

        return (count or 0) > 0


    async def close(self) -> None:
        """
        closes the connection pool gracefully
        """

        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("SemanticStore connection pool closed!")