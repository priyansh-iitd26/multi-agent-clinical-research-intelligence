# the supervisor is the orchestrator of the entire agent graph
# it has only two jobs ->

# job 1 -> ROUTE (supervisor_route) :
# read the incoming task, prepare the shared state, and hand off
# to all six specialist agents simultaneously

# job 2 -> COMPILE (supervisor_compile) :
# after all six specialists finish, read every signal they found,
# rank them by priority, and write one final intelligence brief

# what the supervisor agent does not do ->
# the supervisor does NOT analyse any studies itself
# it does NOT call any tools
# it does NOT search the database
# its only job is coordination - routing work and compiling output
# this separation keeps each component focused and testable

import uuid

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import GraphState, SignalOutput

from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

# llm instance
llm = ChatOpenAI(
    model = settings.openai_chat_model,
    temperature = 0.1,
    # temperature controls how creative vs deterministic the output is
    # 0.0 => completely deterministic (same input -> same output always)
    # 1.0 => very creative, random, unpredictable
    # 0.1 => mostly deterministic with slight variation
    # we use 0.1 because intelligence briefs should be factual and
    # consistent - not creative

    api_key = settings.openai_api_key
)


# function 1: supervisor_route

# this is the FIRST node in the LangGraph graph
# it runs before any specialist agent
# its job is to initialize the run and prepare state for the specialists

async def supervisor_route(state: GraphState) -> dict:
    """
    the entry point of every analysis run

    1. generates a unique run ID for this analysis session
    2. logs what task is being investigated
    3. returns an updated state that all specialists will receive

    the supervisor does NOT need to read the task and decide which
    specialists to activate - we always run ALL six specialists in
    parallel for every task. This is by design:
    - different agents may find different signals in the same task
    - running all six costs the same time as running one (parallel)
    - we never miss a signal type by selectively routing

    every LangGraph node must:
    - accept: the current GraphState
    - return: a dict of ONLY the fields that changed

    LangGraph automatically merges the returned dict into the full state

    args:
        state: the current GraphState from LangGraph

    returns:
        dict with updated run_id, agents_activated, and signals fields
    """

    run_id = str(uuid.uuid4())

    logger.info(
        f"Supervisor routing! | "
        f"run_id = {run_id} | "
        f"task = '{state.get('task', '')[:80]}'"
        # state.get('task', '') safely reads the task from state
    )

    return {
        "run_id": run_id,

        "agents_activated": [],
        # starting with empty list - each specialist agent will add its name
        # to this list when it runs 
        # by the end, this list shows exactly which agents were activated

        "signals": [],
        # starting with empty signals list - each specialist appends
        # its found signals to this list
        # because GraphState uses add_messages pattern for signals,
        # LangGraph merges, rather than replaces

        "run_complete": False,
        # False => run is in progress
        # supervisor_compile sets this to True when done

        "error_log": []
        # empty error log at start - agents write errors here
        # if something goes wrong during their run
    }


# function 2: supervisor_compile

# this is the LAST node before END in the LangGraph graph
# it runs after all six specialists have finished
# its job is to read every signal and write the final brief

async def supervisor_compile(state: GraphState) -> dict:
    """
    reads all agent signals and compiles the final intelligence brief

    LangGraph calls this node only after ALL six specialist nodes
    have completed - guaranteed by the graph structure in graph_builder.py

    1. collects all signals from state (from all 6 agents)
    2. separates high-confidence signals from those needing review
    3. uses GPT-4o to write a professional intelligence brief
    4. returns the completed state

    why using gpt-4o to write the final brief ?

    the raw signals are structured data - JSON with fields like
    summary, confidence, nct_id. They are accurate but not readable.
    GPT-4o transforms them into a professional narrative brief that
    a human analyst can read and act on immediately
    """

    signals = state.get("signals", [])
    agents_activated = state.get("agents_activated", [])
    task = state.get("task", "")

    logger.info(
        f"Supervisor compiling brief! | "
        f"signals = {len(signals)} | "
        f"num_agents_activated = {len(agents_activated)}"
    )

    if not signals:
        # no signals found by any agent - return a clean summary
        # this can happen when the task finds no issues in the data
        logger.info("No signals found - returning clean brief...")
        return {
            "final_brief":  (
                "**EXECUTIVE SUMMARY:** Analysis complete. "
                "No significant research integrity signals were detected "
                "for the specified task and study set."
            ),
            "run_complete": True,
            "agents_activated": agents_activated
        }

    # separating signals by review status
    high_confidence_signals = [
        s for s in signals
        if s.get("confidence", 0) >= 0.6
    ]

    review_signals = [
        s for s in signals
        if s.get("confidence", 0) < 0.6
    ]

    # formatting signals for gpt-4o
    signals_text = _format_signals_for_llm(signals)

    # prompts for gpt-4o
    system_prompt = """You are the Chief Intelligence Officer of this clinical trial research integrity system. 
Your job is to compile
a professional executive intelligence brief from the signals generated
by specialist AI agents.

BRIEF FORMAT:
1. EXECUTIVE SUMMARY - 2-3 sentences summarising the most critical findings
2. SIGNALS BY PRIORITY - each signal as a numbered item with:
   - What was found
   - Why it matters
   - What action to take
3. SIGNALS REQUIRING HUMAN REVIEW - list any low-confidence signals
4. PIPELINE HEALTH - note any errors or issues during the run

TONE: Professional, factual, actionable. Write as if briefing a
senior compliance officer or investigative journalist.
Be specific — include NCT IDs, sponsor names, and exact timeframes.
"""

    human_prompt = f"""
ANALYSIS TASK: {task}

SIGNALS FOUND BY AGENTS:
{signals_text}

HIGH CONFIDENCE SIGNALS: {len(high_confidence_signals)}
SIGNALS REQUIRING REVIEW: {len(review_signals)}
AGENTS ACTIVATED: {', '.join(agents_activated)}

Please compile the final intelligence brief now.
"""
    
    # calling gpt-4o
    try:
        response = await llm.ainvoke(
            [
                SystemMessage(content = system_prompt),
                HumanMessage(content = human_prompt)
            ]
            # ainvoke() is the async version of invoke()
            # we await it because it makes a network call to OpenAI
        )

        final_brief = response.content

        logger.info(
            f"Brief compiled successfully! | "
            f"signals_included = {len(signals)} | "
            f"brief_length = {len(final_brief)} chars"
        )

    except Exception as e:
        # if gpt-4o call fails, fall back to a structured plain-text brief
        # the run should not crash just because the LLM had an issue
        logger.error(f"LLM brief compilation failed! | error = {e}")

        final_brief = _fallback_brief(signals, agents_activated, task)
        # _fallback_brief() generates a basic brief from the raw signal
        # data without using gpt-4o

    return {
        "final_brief": final_brief,
        "run_complete": True,
        "total_signals": len(signals),
        "signals_requiring_review": len(review_signals),
        "agents_activated": agents_activated
    }


# private helper : _format_signals_for_llm

def _format_signals_for_llm(signals: list[SignalOutput]) -> str:
    """
    converts a list of signal dicts into a clean, readable text block
    that gpt-4o can effectively summarise into the final brief

    why format before sending to gpt-4o ?

    raw signal dicts are JSON - full of curly braces and quotes
    gpt-4o works better with plain, labelled text than raw JSON
    formatting the signals into clear sections may produce better briefs
    """

    if not signals:
        return "No signals generated!"

    lines = []

    for i, signal in enumerate(signals, start = 1):

        lines.append(f"SIGNAL {i}:")
        lines.append(f" Agent: {signal.get('agent', 'unknown')}")
        lines.append(f" Type: {signal.get('signal_type', 'unknown')}")
        lines.append(f" NCT ID: {signal.get('nct_id', 'N/A')}")
        lines.append(f" Confidence: {signal.get('confidence', 0.0):.2f}")

        lines.append(f" Summary: {signal.get('summary', '')}")
        lines.append("")

    return "\n".join(lines)


# private helper : _fallback_brief

def _fallback_brief(
    signals:          list,
    agents_activated: list,
    task:             str,
) -> str:
    """
    generates a basic structured brief without using gpt-4o

    called only when the LLM call fails - ensures the API always returns
    something useful even if OpenAI is down or rate-limited

    the output is less polished than the GPT-4o brief but contains
    all the factual information the caller needs
    """

    lines = [
        "**EXECUTIVE SUMMARY:**",
        f"Analysis complete. {len(signals)} signal(s) detected.",
        "",
        "**SIGNALS BY PRIORITY:**",
        ""
    ]

    for i, signal in enumerate(signals, start=1):
        lines.append(
            f"{i}. **{signal.get('nct_id', 'Unknown')} "
            f"- {signal.get('signal_type', 'Unknown')}:**"
        )
        lines.append(f" {signal.get('summary', 'No summary available.')}")
        lines.append(
            f" Confidence: {signal.get('confidence', 0.0):.2f} | "
            f"Agent: {signal.get('agent', 'unknown')}"
        )
        lines.append("")

    lines.append(f"**AGENTS ACTIVATED:** {', '.join(agents_activated)}")
    lines.append(
        "\n*Note: This brief was generated without LLM assistance "
        "due to a temporary error. Please review raw signals directly.*"
    )

    return "\n".join(lines)