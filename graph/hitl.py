# HITL => Human In The Loop
# when an agent's confidence is below its threshold, the signal goes
# into a review queue instead of being saved directly to the signals table
# a human analyst reviews it, approves/rejects/edits it
# rejections feed back into procedural memory - the learning loop

import json
import uuid
import asyncpg
from datetime import datetime
from typing import Any

from memory.procedural_store import ProceduralStore
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


# confidence thresholds per agent
# signals below these go to the human review queue
CONFIDENCE_THRESHOLDS = {
    "broken_promises_agent": 0.60,
    "missing_results_agent": 0.65,
    "track_record_agent": 0.70,
    "pattern_finder_agent": 0.65,
    "side_effect_agent": 0.55,
    "timeline_agent": 0.60
}


class HITL:
    """
    routes signals to either direct save (in signals table) or human review queue (in hitl_reviews table)
    based on the agent's confidence score

    high confidence  -> saved directly to signals table
    low confidence   -> sent to review queue for human review
    """

    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        self._procedural_store = ProceduralStore()

    async def _ensure_pool(self) -> None:
        if self._pool is not None:
            return
        
        self._pool = await asyncpg.create_pool(
            host = settings.db_host,
            port = settings.db_port,
            database = settings.db_name,
            user = settings.db_user,
            password = settings.db_password,
            min_size = 1,
            max_size = 5
        )

    async def process_signal(self, signal: dict[str, Any]) -> dict[str, Any]:
        """
        routes one signal through HITL
        returns a dict with keys: "action", "signal_id", "queue_id"
        action is either "saved_directly" or "sent_to_review"
        """

        await self._ensure_pool()

        agent_name = signal.get("agent", "unknown")
        confidence = signal.get("confidence", 0.0)
        threshold = CONFIDENCE_THRESHOLDS.get(agent_name, 0.65)

        if confidence >= threshold:
            # high confidence -> save directly to signals table
            signal_id = await self._save_signal(signal)
            logger.info(
                f"Signal saved directly! | "
                f"agent = {agent_name} | confidence = {confidence:.2f}"
            )
            return {"action": "saved_directly", "signal_id": signal_id}
        else:
            # low confidence -> send to review queue
            queue_id = await self._send_to_review_queue(signal)
            logger.info(
                f"Signal sent to review | "
                f"agent = {agent_name} | confidence = {confidence:.2f} | "
                f"threshold = {threshold}"
            )
            return {"action": "sent_to_review", "queue_id": queue_id}

    async def _save_signal(self, signal: dict[str, Any]) -> str:
        """
        inserts a high-confidence signal into the signals table
        """

        signal_id = str(uuid.uuid4())

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            await conn.execute(
                """
                INSERT INTO signals (
                    signal_id, nct_id, agent, signal_type,
                    summary, evidence, confidence, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (signal_id) DO NOTHING
                """,
                signal_id,
                signal.get("nct_id", ""),
                signal.get("agent", ""),
                signal.get("signal_type", ""),
                signal.get("summary", ""),
                json.dumps(signal.get("evidence", [])),
                signal.get("confidence", 0.0),
                "approved"
            )

        return signal_id

    async def _send_to_review_queue(self, signal: dict[str, Any]) -> str:
        """
        inserts a low-confidence signal into the hitl_reviews table
        status is "pending" - waiting for human decision
        """

        queue_id  = str(uuid.uuid4())
        signal_id = str(uuid.uuid4())

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            # first saving the signal itself
            await conn.execute(
                """
                INSERT INTO signals (
                    signal_id, nct_id, agent, signal_type,
                    summary, evidence, confidence, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                signal_id,
                signal.get("nct_id", ""),
                signal.get("agent", ""),
                signal.get("signal_type", ""),
                signal.get("summary", ""),
                json.dumps(signal.get("evidence", [])),
                signal.get("confidence", 0.0),
                "pending"
            )

            # creating the review queue (hitl_reviews) entry
            await conn.execute(
                """
                INSERT INTO hitl_reviews (
                    review_id, signal_id, decision
                )
                VALUES ($1, $2, $3)
                """,
                queue_id,
                signal_id,
                "pending"
            )

        return queue_id


    async def process_human_decision(
        self,
        queue_id: str,
        decision: str,   # "approve", "reject", "edit"
        reviewer: str,
        rejection_reason: str = "",
        edit_summary: str = ""
    ) -> dict[str, Any]:
        """
        processes a human reviewer's decision on a queued signal

        approve -> signal status becomes "approved"
        edit -> signal summary updated, status becomes "approved"
        reject -> signal status becomes "rejected" AND rejection reason
                  is written to procedural memory (the learning loop)
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            # getting the signal_id from the review queue entry
            row = await conn.fetchrow(
                "SELECT signal_id FROM hitl_reviews WHERE review_id = $1",
                queue_id
            )

            if not row:
                return {"success": False, "error": "Review not found"}

            signal_id  = row["signal_id"]
            new_status = "approved" if decision in ("approve", "edit") else "rejected"

            # updating the signal status
            await conn.execute(
                "UPDATE signals SET status = $1 WHERE signal_id = $2",
                new_status, signal_id
            )

            # updating the review record with the decision
            await conn.execute(
                """
                UPDATE hitl_reviews
                SET decision = $1, reviewer = $2,
                    rejection_reason = $3, edit_summary = $4,
                    reviewed_at = NOW()
                WHERE review_id = $5
                """,
                decision, reviewer, rejection_reason, edit_summary, queue_id
            )

            # if edited -> update the signal summary with the correction
            if decision == "edit" and edit_summary:
                await conn.execute(
                    "UPDATE signals SET summary = $1 WHERE signal_id = $2",
                    edit_summary, signal_id
                )

        # learning loop

        # if the human REJECTED the signal, write the reason to
        # procedural memory so the agent reasons differently next time

        if decision == "reject" and rejection_reason:

            agent_row = await self._get_agent_for_signal(signal_id)

            if agent_row:
                await self._procedural_store.update_from_feedback(
                    agent_name=agent_row,
                    rejection_reason=rejection_reason,
                )
                logger.info(
                    f"Procedural memory updated from rejection! | "
                    f"agent = {agent_row}"
                )

        return {
            "success":   True,
            "decision":  decision,
            "signal_id": signal_id,
            "queue_id":  queue_id,
        }


    async def _get_agent_for_signal(self, signal_id: str) -> str | None:
        """
        returns the agent name for a given signal_id
        """

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            return await conn.fetchval(
                "SELECT agent FROM signals WHERE signal_id = $1",
                signal_id
            )


    async def get_review_queue(self) -> list[dict]:
        """
        returns all signals currently pending human review
        """

        await self._ensure_pool()

        assert self._pool is not None
        async with self._pool.acquire() as conn:

            rows = await conn.fetch(
                """
                SELECT
                    hr.review_id,
                    hr.signal_id,
                    s.agent,
                    s.signal_type,
                    s.summary,
                    s.confidence,
                    s.nct_id,
                    hr.decision,
                    hr.reviewed_at
                FROM hitl_reviews hr
                JOIN signals s ON hr.signal_id = s.signal_id
                WHERE hr.decision = 'pending'
                ORDER BY s.confidence ASC
                -- Show lowest confidence signals first —
                -- these need the most urgent review.
                """
            )

        return [dict(row) for row in rows]


    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None