import json
import logging
import time
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
from evaluation.fixtures import FixtureManager
from evaluation.reporting import ReportGenerator
from evaluation.scorers import SchemaFidelityScorer, RecallAtKScorer, LLMJudgeScorer
from evaluation.judge import JudgeOrchestrator, OllamaJudgeProvider

class EvalStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"

@dataclass
class EvaluationCase:
    case_id: str
    component: str
    category: str # golden, edge, negative, boundary
    input_fixture: str
    expected_output: Optional[Dict[str, Any]] = None
    metrics: List[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class MetricResult:
    metric_name: str
    score: Union[float, int, str]
    status: EvalStatus
    rationale: str
    evidence: Optional[List[Dict[str, Any]]] = None

@dataclass
class CaseResult:
    case_id: str
    component: str
    results: List[MetricResult]
    overall_status: EvalStatus
    execution_time: float
    metadata: Dict[str, Any] = None

class EvalRunner:
    """
    Core execution engine for offline evaluation.
    Orchestrates the loading of fixtures, execution of components,
    and invocation of scorers/judges.
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("eval_runner")
        self.fixture_mgr = FixtureManager(config.get("fixture_dir", "evaluation/fixtures"))
        self.report_gen = ReportGenerator()
        self.judge_orch = JudgeOrchestrator(OllamaJudgeProvider())

    def load_case(self, case_id: str) -> EvaluationCase:
        with open(f"{self.config['fixture_dir']}/manifest.json", 'r') as f:
            manifest = json.load(f)
        for case in manifest["cases"]:
            if case["case_id"] == case_id:
                return EvaluationCase(**case)
        raise ValueError(f"Case {case_id} not found in manifest.")

    async def run_component_eval(self, component_name: str, case: EvaluationCase):
        start_time = time.time()
        # Mocking the actual component execution for the framework verification
        # In real use, this would call the actual agent/pipeline code
        candidate_output = {}
        if component_name == "ingestion":
            candidate_output = self.fixture_mgr.load_raw_mock(case.input_fixture.split("/")[-1])
        elif "agent" in component_name:
            candidate_output = {"signal_detected": True, "reasoning": "Mocked agent signal"}
        elif component_name == "supervisor":
            candidate_output = {"brief": "Mocked supervisor brief"}

        metric_results = []
        for metric in (case.metrics or []):
            if metric == "schema_fidelity":
                scorer = SchemaFidelityScorer()
                res = scorer.score(candidate_output, case.expected_output)
                metric_results.append(MetricResult(metric, res.score, EvalStatus.PASS if res.status == "PASS" else EvalStatus.FAIL, res.rationale))
            elif metric == "grounding_score":
                # In a real run, this would use self.judge_orch
                metric_results.append(MetricResult(metric, 4, EvalStatus.PASS, "Mocked grounding score 4/4"))
            elif metric == "synthesis_utility":
                metric_results.append(MetricResult(metric, 4, EvalStatus.PASS, "Mocked synthesis score 4/4"))

        overall_status = EvalStatus.PASS if all(m.status == EvalStatus.PASS for m in metric_results) else EvalStatus.FAIL

        return CaseResult(
            case_id=case.case_id,
            component=component_name,
            results=metric_results,
            overall_status=overall_status,
            execution_time=time.time() - start_time
        )

    def calculate_component_score(self, results: List[CaseResult]) -> float:
        if not results: return 0.0
        scores = [r.results[0].score for r in results if r.results]
        return sum(scores) / len(scores) if scores else 0.0

    def generate_report(self, results: List[CaseResult], format: str = "markdown"):
        if format == "markdown":
            self.report_gen.generate_markdown_summary(results, "evaluation_report.md")
        else:
            self.report_gen.generate_json_report(results, "evaluation_report.json")
