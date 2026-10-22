import json
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

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

    def load_case(self, case_id: str) -> EvaluationCase:
        # Implementation will load from fixtures/manifest.json
        pass

    async def run_component_eval(self, component_name: str, case: EvaluationCase):
        # Logic to route to specific evaluator
        pass

    def calculate_component_score(self, results: List[CaseResult]) -> float:
        # Aggregation logic (averages, critical gates)
        pass

    def generate_report(self, results: List[CaseResult], format: str = "markdown"):
        # Report generation logic
        pass
