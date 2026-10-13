from pydantic import BaseModel, Field
from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from config.settings import settings

class JudgeScore(BaseModel):
    groundedness: int = Field(description="Score 1-5: Are claims strictly supported by retrieved signals without hallucination?")
    relevance: int = Field(description="Score 1-5: Does the summary directly address the user's task?")
    clarity: int = Field(description="Score 1-5: Is the brief actionable, clear, and well-structured?")
    reasoning: str = Field(description="Brief explanation for each score")

class JudgeResult(BaseModel):
    scores: JudgeScore
    overall_pass: bool = Field(description="True if total score >= 12 and no single score < 3")

# NEW: Model to structure the final aggregate metrics for your report
class EvaluationMetrics(BaseModel):
    total_evaluated: int
    passed_count: int
    overall_pass_rate_percentage: float
    average_groundedness: float
    average_relevance: float
    average_clarity: float

class LLMJudge:
    def __init__(self):
        # Note: In a real offline environment, this would use a local model wrapper.
        # For the current project settings, we use the configured HF model.
        self.llm = ChatOpenAI(
            model=settings.huggingface_chat_model,
            base_url=settings.huggingface_base_url,
            api_key=settings.huggingface_api_key,
            temperature=0
        ).with_structured_output(JudgeResult)

    async def evaluate_brief(self, task: str, signals: list, brief: str) -> JudgeResult:
        system_prompt = """You are a Senior Clinical Research Auditor.
        Evaluate the provided Intelligence Brief based on the raw signals found.

        CRITERIA:
        1. Groundedness: Penalize any claim not explicitly present in the signals.
        2. Relevance: Penalize summaries that ignore the user's specific task.
        3. Clarity: Penalize jargon, vagueness, or poor formatting.
        """

        human_prompt = f"""
        TASK: {task}
        RAW SIGNALS: {signals}
        GENERATED BRIEF: {brief}

        Provide a structured evaluation.
        """

        return await self.llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

    # NEW: Calculates mathematical averages from a list of JudgeResults
    def calculate_metrics(self, judge_results: List[JudgeResult]) -> EvaluationMetrics:
        if not judge_results:
            return EvaluationMetrics(
                total_evaluated=0, passed_count=0, overall_pass_rate_percentage=0.0,
                average_groundedness=0.0, average_relevance=0.0, average_clarity=0.0
            )

        total = len(judge_results)
        passed = sum(1 for r in judge_results if r.overall_pass)
        
        avg_groundedness = sum(r.scores.groundedness for r in judge_results) / total
        avg_relevance = sum(r.scores.relevance for r in judge_results) / total
        avg_clarity = sum(r.scores.clarity for r in judge_results) / total
        
        return EvaluationMetrics(
            total_evaluated=total,
            passed_count=passed,
            overall_pass_rate_percentage=round((passed / total) * 100, 2),
            average_groundedness=round(avg_groundedness, 2),
            average_relevance=round(avg_relevance, 2),
            average_clarity=round(avg_clarity, 2)
        )

    async def evaluate_batch(self, cases: List[Dict[str, Any]], max_concurrency: int = 2) -> Tuple[List[JudgeResult], EvaluationMetrics]:
        import asyncio

        # Semaphore limits outbound TCP connections to avoid WinError 1225
        semaphore = asyncio.Semaphore(max_concurrency)

        async def sem_eval(case):
            async with semaphore:
                # Retry loop for network resilience
                for attempt in range(3):
                    try:
                        result = await self.evaluate_brief(
                            task=case.get("task", ""),
                            signals=case.get("signals", []),
                            brief=case.get("brief", "")
                        )
                        return result
                    except Exception as e:
                        if attempt < 2:
                            await asyncio.sleep(1.5 * (attempt + 1))  # Backoff delay before retry
                        else:
                            print(f"Warning: Judge call failed after 3 attempts for task '{case.get('task', '')[:30]}...': {e}")
                            return JudgeResult(
                                scores=JudgeScore(groundedness=1, relevance=1, clarity=1, reasoning=f"Connection Error: {e}"),
                                overall_pass=False
                            )

        tasks = [sem_eval(case) for case in cases]
        results = await asyncio.gather(*tasks)
        metrics = self.calculate_metrics(results)
        
        return results, metrics