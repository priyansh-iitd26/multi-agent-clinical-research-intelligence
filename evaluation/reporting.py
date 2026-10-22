import json
from typing import List, Dict, Any
from evaluation.eval_runner import EvalStatus

class ReportGenerator:
    """
    Generates machine-readable and human-readable reports from evaluation results.
    """
    def generate_json_report(self, results: List[Any], output_path: str):
        # Implementation: dump results to JSON
        pass

    def generate_markdown_summary(self, results: List[Any], output_path: str):
        # Implementation: Create summary table with scores and failures
        pass

    def analyze_failures(self, results: List[Any]) -> Dict[str, Any]:
        # Implementation: Classify failures (R-FAIL, G-FAIL, S-FAIL, T-FAIL)
        pass
