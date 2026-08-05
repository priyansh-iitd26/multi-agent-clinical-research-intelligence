# this file gives the agents the ability to learn HOW to reason
# better over time - based on feedback from human reviewers

# procedural memory analogue ->
# think of a junior doctor who just started working
# on day 1, they follow a basic checklist when diagnosing patients
# over time, senior doctors correct them:
# "when a patient has BOTH symptom A and symptom B together,
# always check for condition X first - not condition Y"
# the junior doctor writes this correction into their personal
# rulebook, next time they see that combination, they follow
# the updated rule automatically

# that is exactly what procedural memory does for our agents
# each agent starts with a set of DEFAULT reasoning rules
# whenever a human reviewer REJECTS a signal and explains why,
# that rejection reason gets written into the agent's rulebook
# as a new rule - permanently changing how it reasons
#
# EPISODIC vs PROCEDURAL
# episodic memory => WHAT happened in past sessions
#                      "I found missing results for NCT04788680"
#                      like a diary - records of past events
#
# procedural memory => HOW to reason - the rules themselves
#                       "When a trial was terminated early,
#                        missing results are expected - not a violation"
#                        like a rulebook - guidelines for behaviour

# 1. agent generates a signal -> "NCT04788680 has missing results"
# 2. human reviewer looks at it
# 3. human REJECTS it with reason:
#      "this trial was terminated early due to COVID -
#       missing results for terminated trials are expected"
# 4. that reason gets written into the agent's procedures table
#    as a new rule: "check if trial was TERMINATED before flagging
#      missing results - terminated trials are exempted"
# 5. next time the agent runs, it LOADS its procedures first
# 6. it applies the new rule and correctly skips terminated trials

# one human correction -> agent reasons differently FOREVER after
# this is the learning loop that makes our agentic system smarter over time
#
# where procedures are stored :
#   in Cloud SQL instance, in the "procedures" table

import asyncpg
import json
from datetime import datetime, UTC

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

# what each column means in procedures table ->
# procedure_id  -> unique ID for this rule
# agent_name    -> which agent owns this rule
# rule_text     -> the actual reasoning rule in plain English
# rule_type     -> "default" (built-in) or "learned" (from HITL)
# source        -> "default" (initial) or "hitl_rejection" (from human)
# created_at    -> when this rule was first created
# updated_at    -> when this rule was last modified

# DEFAULT RULES : every agent starts with these

# these are the built-in reasoning rules that every agent loads
# at the very start of first session - before doing any analysis

# when a human rejects a signal and explains why, a new rule is
# added to the database alongside these defaults
# on the next run, the agent loads both the defaults and any learned rules

DEFAULT_RULES = {
    # each key is an agent name
    # each value is a list of rule strings for that agent
    # these strings are in plain english -> the agent reads them directly
    # as part of its system prompt before it starts reasoning

    "missing_results_agent": [
        "Flag a study as missing results ONLY if status is COMPLETED "
        "and results_posted is False and more than 12 months have "
        "passed since the completion date.",

        "Do NOT flag studies with status TERMINATED as missing results. "
        "Terminated trials are not legally required to post results "
        "in all circumstances — termination often means the study "
        "was stopped early and has incomplete data.",
    
        "If enrollment was zero or very low (under 10 participants), "
        "note this in the signal but reduce confidence to 0.5. "
        "A study that never really started may not have reportable results.",

        "Always check the sponsor's track record before assigning "
        "a confidence score. A first-time missing result from a "
        "historically compliant sponsor warrants lower confidence "
        "than the same finding from a repeat offender.",
    ],

    "broken_promises_agent": [
        "Flag outcome switching ONLY when the PRIMARY outcome changes "
        "after enrollment has begun. Changes to secondary outcomes "
        "are less concerning and should not trigger a HIGH confidence signal.",

        "A change in outcome MEASUREMENT METHOD (how it is measured) "
        "is different from a change in the outcome itself. "
        "Method changes may be legitimate protocol improvements — "
        "flag them at MEDIUM confidence, not HIGH.",

        "If a protocol amendment was filed BEFORE enrollment began, "
        "the outcome change is less suspicious — the study had not "
        "yet collected data that could have influenced the change. "
        "Assign MEDIUM confidence in this case.",

        "Always note the date of the change relative to the "
        "enrollment start date — this timing is the most important "
        "factor in assessing whether outcome switching is intentional.",
    ],

    "track_record_agent": [
        "A credibility score below 0.6 should trigger a LOW_CREDIBILITY "
        "signal. Between 0.6 and 0.75 is concerning but not alarming — "
        "note it in the analysis but do not generate a signal.",

        "Weight recent behaviour more heavily than old behaviour. "
        "A sponsor with 5 violations in the last 2 years is more "
        "concerning than one with 10 violations spread over 20 years.",

        "If a sponsor has fewer than 3 studies in our database, "
        "reduce confidence to 0.5. We do not have enough data to "
        "make a reliable judgment about their track record.",

        "Always distinguish between a sponsor's PRIMARY studies "
        "(where they are the lead sponsor) and COLLABORATIVE studies "
        "(where they are a secondary party). Hold them more accountable "
        "for their primary studies.",
    ],

    "pattern_finder_agent": [
        "A cross-study pattern requires at least 3 studies to be "
        "meaningful. Two studies with similar issues may be coincidence. "
        "Three or more is a pattern worth flagging.",

        "When multiple companies are testing the same drug for the "
        "same condition, check whether any of them have hidden "
        "negative results from previous studies in our database.",

        "A drug that failed Phase 2 for condition A but is being "
        "retried in Phase 2 for condition B is worth flagging — "
        "especially if the mechanism of action is the same.",

        "Patterns across the same SPONSOR are more actionable than "
        "patterns across different sponsors. Same-sponsor patterns "
        "suggest systemic issues, not coincidence.",
    ],

    "side_effect_agent": [
        "A safety discrepancy between the official filing and a "
        "published paper is only meaningful if the paper was published "
        "AFTER the trial completed — not during it.",

        "Look specifically for cases where the filing says "
        "'no serious adverse events' but published papers mention "
        "hospitalisations, discontinuations, or deaths. "
        "This is the highest-priority safety signal.",

        "If the discrepancy is minor (e.g. different terminology "
        "for the same event), assign LOW confidence. "
        "If the discrepancy involves severity (mild vs serious), "
        "assign HIGH confidence.",

        "Always note whether the paper's authors are the same as "
        "the trial's investigators. Independent authors are more "
        "credible than sponsor-employed investigators.",
    ],

    "timeline_agent": [
        "Flag a delay ONLY if it exceeds 180 days beyond the "
        "stated completion date AND no amendment was filed explaining "
        "the extension. A silent delay is more suspicious than "
        "a disclosed one.",

        "COVID-19 is a legitimate reason for delays between "
        "March 2020 and December 2022. Do not flag delays in this "
        "period as suspicious without additional evidence.",

        "A study that is recruiting past its stated completion date "
        "may simply have underestimated enrollment time — this is "
        "common and not inherently suspicious. Focus on COMPLETED "
        "studies that are past their results posting deadline.",

        "Always compare the actual completion date against BOTH "
        "the original completion date AND any amended completion "
        "dates. Use the most recent amendment as the baseline.",
    ],
}

# procedural memory class
class ProceduralStore:
    """
    stores and retrieves agent reasoning rules (procedures)

    each agent has its own set of procedures - rules that guide
    how it reasons about clinical trial data
    procedures come in two types:
      1. DEFAULT -> built-in rules the agent always starts with
      2. LEARNED -> rules added when a human reviewer corrects the agent

    usage:
        store = ProceduralStore()

        # load all rules for an agent before it starts reasoning
        rules = await store.get_procedures("missing_results_agent")

        # update rules after a human rejection
        await store.update_from_feedback(
            agent_name="missing_results_agent",
            rejection_reason="Terminated trials should not be flagged"
        )
    """

    def __init__(self):

        # lazy initialization
        self._pool: asyncpg.Pool | None = None

        logger.info("ProceduralStore initialized!")


    # private helper method : _ensure_pool
    async def _ensure_pool(self) -> None:
        """
        creates the database connection pool if it does not exist
        called at the start of every public method - guarantees
        the database is reachable before we try to use it
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
            max_size = 3
        )

        logger.info("ProceduralStore connection pool created!")


    # public core method : initialize_defaults
    async def initialize_defaults(self) -> None:
        """
        inserts the DEFAULT_RULES into the procedures table

        when is this called ?
        once - when our system starts up for the very first time
        after that, the rules are already in the database and
        this method safely skips any that already exist
        (using ON CONFLICT DO NOTHING)

        why not hardcode rules in the agent ?
        because we want rules to be:
            1. stored in the database - persistent across runs
            2. updateable - humans can add new learned rules
            3. visible - the API can return current rules on demand
            4. auditable - we can see how rules changed over time

        if rules were hardcoded in the agent, learned rules would
        disappear every time the application restarted 
        storing them in the database makes them permanent
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            for agent_name, rules in DEFAULT_RULES.items():

                for rule_text in rules:
                    # rules is the list of rule strings for this agent
                    # we insert each rule as a separate row in the table
                    # one row -> one rule 
                    # an agent with 4 rules → 4 rows

                    await conn.execute(
                        """
                        INSERT INTO procedures (
                            agent_name,
                            rule_text,
                            rule_type,
                            source
                        )
                        VALUES ($1, $2, $3, $4)
                        ON CONFLICT DO NOTHING
                        """,
                        agent_name,
                        rule_text,
                        "default",
                        "default",
                    )

        logger.info(
            f"Default procedures initialized! | "
            f"agents = {list(DEFAULT_RULES.keys())}"
        )


    # core method : get_procedures
    async def get_procedures(
        self,
        agent_name: str,
        # which agent's rules to load
        # example: "missing_results_agent"
        # each agent only loads its own rules -
        # an agent should not follow another agent's reasoning rules
    ) -> list[str]:
        """
        returns all the reasoning rules for a specific agent

        this is called at the very start of every agent session -
        before the agent does any analysis 
        ** the agent reads these rules and incorporates them into its system prompt 
        so that GPT-4o reasons according to them

        the returned list contains BOTH:
        - default rules (built-in at startup)
        - learned rules (added from human feedback over time)

        the agent has no idea which rules are default and which
        are learned - it just receives a list and follows all of them
        this is intentional - all rules are equally authoritative
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            rows = await conn.fetch(
                """
                SELECT rule_text
                FROM procedures
                WHERE agent_name = $1
                ORDER BY created_at ASC
                """,
                agent_name
            )

        rules = [row["rule_text"] for row in rows]

        logger.info(
            f"Procedures loaded for agent = {agent_name} ! | "
            f"rules_count = {len(rules)}"
        )

        return rules


    # core method : update_from_feedback
    async def update_from_feedback(
        self,
        agent_name: str,
        rejection_reason: str,
        # the reason the human reviewer gave for rejecting the signal
        # example: "This trial was terminated early due to COVID -
        #           terminated trials are exempt from result posting"
        # this plain-English reason becomes a new reasoning rule
        # the agent will read it at the start of every future session
    ) -> str:
        """
        adds a new learned reasoning rule from a human rejection (HITL)

        THIS IS THE LEARNING LOOP IN ACTION

        when a human reviewer rejects an agent's signal, they explain
        why it was wrong
        that explanation is passed to this method
        this method saves it as a new rule in the procedures table for that agent

        from this point forward, every time this agent runs, it will:
            1. load its procedures (including this new rule)
            2. apply the new rule during its reasoning
            3. avoid making the same mistake again
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:
            
            procedure_id = await conn.fetchval(
                # fetchval() returns a single value — not a list of rows
                # perfect here because INSERT ... RETURNING gives us
                # one value back: the auto-generated procedure_id (uuid)
                """
                INSERT INTO procedures (
                    agent_name,
                    rule_text,
                    rule_type,
                    source
                )
                VALUES ($1, $2, $3, $4)
                RETURNING procedure_id
                """,
                # RETURNING procedure_id means:
                # after inserting the new row, give us back the
                # procedure_id that was auto-generated for it
                # this lets us return the ID to the caller without
                # needing to run a separate SELECT query

                agent_name,
                rejection_reason,
                "learned",
                "hitl_rejection",
            )

        logger.info(
            f"Procedure learned from feedback! | "
            f"agent = {agent_name} | "
            f"rule_preview = '{rejection_reason[:80]}...' | "
            f"procedure_id = {procedure_id}"
        )

        return str(procedure_id)


    # utility method : get_all_procedures
    async def get_all_procedures(
        self,
        agent_name: str,
    ) -> list[dict]:
        """
        returns all procedures for an agent with full metadata

        unlike get_procedures() which just returns rule text strings,
        this method returns the full procedure record including
        the rule type, source, and timestamps

        used by the FastAPI endpoint:
        GET /api/v1/memory/procedures/{agent_name}

        this lets analysts see:
            - what rules the agent currently follows
            - which are built-in vs learned from feedback
            - when each rule was added
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            rows = await conn.fetch(
                """
                SELECT
                    procedure_id,
                    agent_name,
                    rule_text,
                    rule_type,
                    source,
                    created_at
                FROM procedures
                WHERE agent_name = $1
                ORDER BY created_at ASC
                """,
                agent_name
            )

        return [
            {
                "procedure_id": str(row["procedure_id"]),
                "agent_name":   row["agent_name"],
                "rule_text":    row["rule_text"],
                "rule_type":    row["rule_type"],
                "source":       row["source"],
                "created_at":   str(row["created_at"])
            }
            for row in rows
        ]


    # close the connection pool and release db connections
    async def close(self) -> None:
        """
        closes the connection pool gracefully
        """

        if self._pool:

            await self._pool.close()

            self._pool = None

            logger.info("ProceduralStore connection pool closed!")