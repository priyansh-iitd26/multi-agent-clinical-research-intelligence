# finds completed clinical trials that never posted their results
# by US law, sponsors must post results within 12 months
# of the primary completion date - about 30% of completed trials
# never do this - this agent finds every single one in our corpus

# what this agent does step-by-step :
# 1. loads its reasoning procedures from procedural memory
#    (the rules it has learned from human feedback over time)
# 2. searches past episodes - has it seen similar cases before?
# 3. searches the database for completed studies with no results
# 4. for each suspicious study, verifies with a live API call
# 5. checks the sponsor's track record
# 6. generates a signal with confidence score
# 7. updates the sponsor's profile in semantic memory
# 8. saves this session as an episode in episodic memory

# confidence threshold :
# >= 0.65 -> saved directly (high confidence, clear violation)
# <  0.65 -> sent to human review queue (hitl_reviews)

import json
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.tools import tool

from graph.state import GraphState, SignalOutput
from memory.procedural_store import ProceduralStore
from memory.episodic_store import EpisodicStore
from memory.semantic_store import SemanticStore
from tools.search_tools import (
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
)
from tools.clinical_tools import check_results_posted, fetch_study_details
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


AGENT_NAME = "missing_results_agent"
SIGNAL_TYPE = "missing_results"
# signal type label that appears in the signals table
# analysts filter signals by type - this label must be consistent
# across every run so filters always work correctly

AGENT_TOOLS = [
    search_studies_by_meaning,
    search_past_episodes,
    save_episode,
    get_sponsor_profile,
    update_sponsor_profile,
    check_results_posted,
    fetch_study_details
]
# each agent gets only the tools it actually needs
# giving an agent too many tools confuses gpt-4o - it spends
# reasoning capacity deciding which tool to use instead of
# focusing on the actual analysis task
# this focused toolset keeps the agent sharp and efficient


_procedural = ProceduralStore()
# loads reasoning rules for this agent at the start of every run

_episodic   = EpisodicStore()
# saves and searches this agent's past sessions

_semantic   = SemanticStore()
# reads and updates sponsor credibility profiles


_llm = ChatOpenAI(
    model = settings.openai_chat_model,
    temperature = 0.1,
    api_key = settings.openai_api_key
).bind_tools(AGENT_TOOLS)
# .bind_tools() attaches our tool list to the chat model
# now when gpt-4o reasons and decides it needs to search the database,
# it can emit a "tool_call" in its response
# LangGraph sees the tool_call, executes the right function,
# and feeds the result back to the agent automatically
# without .bind_tools(), the agent has no way to access data -
# it would only reason from what is in its system prompt


async def missing_results_node(state: GraphState) -> dict:
    """
    the Missing Results Agent node - called by LangGraph during graph execution

    finds completed clinical trials that never posted results

    returns the updated state with any signals found and this agent's name
    added to agents_activated
    """

    logger.info(f"{AGENT_NAME} | Starting analysis!")

    try:
        # step 1 : loading reasoning procedures for this agent
        # this includes both default rules AND any rules learned
        # from human feedback via the HITL rejection loop
        procedures = await _procedural.get_procedures(AGENT_NAME)

        procedures_text = "\n".join(f"- {r}" for r in procedures)

        # step 2 : system prompt
        system_prompt = f"""You are the Missing Results Agent for this 
clinical trial research integrity intelligence system.

YOUR MISSION:
Find completed clinical trials that have never posted their results
to ClinicalTrials.gov, violating federal law (FDAAA 801).
By law, results must be posted within 12 months of primary completion.

YOUR REASONING RULES (follow these exactly):
{procedures_text}

YOUR WORKFLOW:
1. Search past episodes to see if you have investigated similar cases
2. Search the database for completed studies with missing results
3. For each suspicious study, verify the current status with a live API call
4. Check the sponsor's track record using get_sponsor_profile
5. Generate a signal with confidence score based on evidence strength
6. Update the sponsor profile with your findings
7. Save this session as an episode before finishing

CONFIDENCE SCORING GUIDE:
- 0.9+ : Completed 5+ years ago, zero results, repeat offender sponsor
- 0.8  : Completed 2-5 years ago, zero results, known non-compliant sponsor
- 0.7  : Completed 1-2 years ago, zero results, average sponsor
- 0.6  : Completed 1 year ago exactly, borderline timing
- Below 0.6: Uncertain — send to human review

OUTPUT FORMAT for each signal found:
Return a JSON block exactly like this:
{{
  "nct_id": "NCT_ID_HERE",
  "signal_type": "missing_results",
  "summary": "Plain English description of what you found",
  "evidence": ["key fact 1", "key fact 2", "key fact 3"],
  "confidence": 0.85
}}

If you find no signals, say "NO_SIGNALS_FOUND" clearly.
"""
        # system prompt is gpt-4o's entire instruction set
        # it defines the agent's role, its rules, its workflow,
        # and the exact output format we need
        # the more precise and structured this prompt is,
        # the more reliable the agent's output will be

        task = state.get("task", "Find completed trials with missing results")

        nct_ids = state.get("nct_ids", [])

        human_message = f"""
ANALYSIS TASK: {task}

SPECIFIC STUDIES TO CHECK: {nct_ids if nct_ids else "Search broadly — no specific studies provided"}

Begin your investigation now. Use your tools to search for completed
studies with missing results. Generate signals for every violation you find.
"""

        # step 3 : agent reasoning loop
        messages = [
            SystemMessage(content = system_prompt),
            HumanMessage(content = human_message)
        ]
        # start the conversation with system instructions and the task
        # LangGraph will extend this list as the agent calls tools
        # and receives tool results - building up the full conversation

        signals_found = []
        # accumulates all signals this agent generates during this run

        max_iterations = 10
        # safety cap to prevent infinite loops if the agent keeps
        # calling tools without reaching a conclusion
        # 10 iterations is more than enough for thorough analysis

        for iteration in range(max_iterations):
            # the agent reasoning loop - each iteration is one LLM call
            # the agent reads the conversation, decides what to do,
            # either calls a tool or writes its final answer

            response = await _llm.ainvoke(messages)
    
            # response.content => text the agent wrote
            # response.tool_calls => tools the agent wants to call

            messages.append(AIMessage(content=response.content or ""))
            # adding the agent's response to the conversation history
            # this is how LangGraph maintains conversation context -
            # every message is appended so the agent always sees
            # the full history of what has been said and done

            if not response.tool_calls:
                # agent wrote a final answer without calling any more tools
                # this means it has finished its analysis
                # parse the response for signal JSON blocks
                logger.info(
                    f"{AGENT_NAME} | Analysis completed! | iteration = {iteration+1}"
                )
                signals_found = _parse_signals(str(response.content), AGENT_NAME)
                break

            # executing tool calls
            for tool_call in response.tool_calls:
                tool_result = await _execute_tool(tool_call, AGENT_TOOLS)
                # running the tool the agent requested and get the result
                # _execute_tool() finds the right tool function by name
                # and calls it with the arguments gpt-4o specified

                messages.append(
                    HumanMessage(
                        content = f"Tool result for {tool_call['name']}:\n{tool_result}"
                    )
                )

        # step 4 : save to episodic store
        episode_content = (
            f"Task: {task}. "
            f"Found {len(signals_found)} missing results signals. "
            f"Signals: {[s.get('nct_id') for s in signals_found]}"
        )

        await _episodic.save_episode(
            agent_name = AGENT_NAME,
            episodic_content = episode_content,
            outcome = "signal_generated" if signals_found else "no_signal"
        )
        # saving to episodic memory so future runs can search:
        # "have I found missing results from this type of sponsor before?"

        logger.info(
            f"{AGENT_NAME} | Completed! | signals_found = {len(signals_found)}"
        )

        # step 5 : returning updated state
        current_signals = state.get("signals", [])
        current_activated = state.get("agents_activated", [])

        return {
            "signals": current_signals + signals_found,
            "agents_activated": current_activated + [AGENT_NAME]
        }

    except Exception as e:

        logger.error(f"{AGENT_NAME} | Error occured! | {e}")
        error_log = state.get("error_log", [])

        return {
            "error_log": error_log + [f"{AGENT_NAME}: {str(e)}"],
            "agents_activated": state.get("agents_activated", []) + [AGENT_NAME]
        }


def _parse_signals(response_text: str, agent_name: str) -> list[SignalOutput]:
    """
    extracts signal JSON blocks from the agent's text response

    gpt-4o writes signals as JSON blocks in its response text
    this function finds and parses every JSON block

    why parse from text ?
    
    we could ask gpt-4o to return structured JSON directly
    but agents need to explain their reasoning in plain text too -
    the text before and after the JSON contains valuable context
    for debugging and audit trails
    parsing JSON from mixed text gives us both
    """

    signals = []

    if not response_text or "NO_SIGNALS_FOUND" in response_text:
        return signals

    import re

    json_pattern = re.compile(r'\{[^{}]*"signal_type"[^{}]*\}', re.DOTALL)

    matches = json_pattern.findall(response_text)

    for match in matches:
        try:
            signal_data = json.loads(match)

            signal: SignalOutput = {
                "agent": agent_name,
                "signal_type": signal_data.get("signal_type", SIGNAL_TYPE),
                "nct_id": signal_data.get("nct_id", ""),
                "summary": signal_data.get("summary", ""),
                "evidence": signal_data.get("evidence", []),
                "confidence": float(signal_data.get("confidence", 0.5))
            }

            signals.append(signal)

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Could not parse signal JSON! | error = {e}")
            continue

    return signals


async def _execute_tool(tool_call: dict, available_tools: list) -> str:
    """
    finds and executes the tool that gpt-4o requested

    LangGraph agents emit tool_calls in their responses - these
    contain the tool name and arguments gpt-4o wants to use
    this function looks up the right tool by name and calls it
    """

    tool_name = tool_call.get("name", "")
    tool_args = tool_call.get("args", {})

    tool_func = None

    for t in available_tools:
        if t.name == tool_name:
            tool_func = t
            break
    # searching through available_tools to find the one with matching name
    # t.name is the tool's registered name from the @tool decorator

    if tool_func is None:
        return f"Error: Tool '{tool_name}' not found in agent's toolset!"

    try:
        result = tool_func.invoke(tool_args)
        # tool_func.invoke() calls the tool with the provided arguments
        # this is LangChain's standard way to call a tool -
        # it handles argument validation and error wrapping
        # returns a string (all our tools return JSON strings)

        return str(result)

    except Exception as e:
        logger.error(f"Tool execution failed! | tool = {tool_name} | error = {e}")
        return f"Error executing tool '{tool_name}': {str(e)}"