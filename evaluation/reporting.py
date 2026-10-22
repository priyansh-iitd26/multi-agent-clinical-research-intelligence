from enum import Enum
from typing import List, Dict, Any

class EvalStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"

class ReportGenerator:
    """
    Generates machine-readable and human-readable reports from evaluation results.
    """


    def generate_json_report(self, results: List[Any], output_path: str):
        with open(output_path, 'w') as f:
            json.dump([asdict(r) if hasattr(r, '__dataclass_fields__') else r for r in results], f, indent=2)

    def generate_markdown_summary(self, results: List[Any], output_path: str):
        with open(output_path, 'w') as f:
            f.write("# Offline Evaluation Summary\n\n")
            f.write("| Case ID | Component | Status | Score |\n")
            f.write("|---|---|---|---|\n")
            for r in results:
                status = r.overall_status.value if hasattr(r.overall_status, 'value') else r.overall_status
                score = "N/A"
                if r.results:
                    score = str(r.results[0].score)
                f.write(f"| {r.case_id} | {r.component} | {status} | {score} |\n")

    def analyze_failures(self, results: List[Any]) -> Dict[str, Any]:
        failures = {"R-FAIL": 0, "G-FAIL": 0, "S-FAIL": 0, "T-FAIL": 0, "OTHER": 0}
        for r in results:
            if r.overall_status != EvalStatus.PASS:
                # Heuristic mapping for this demo
                if "retrieval" in r.component: failures["R-FAIL"] += 1
                elif "agent" in r.component: failures["G-FAIL"] += 1
                elif "supervisor" in r.component: failures["S-FAIL"] += 1
                else: failures["OTHER"] += 1
        return failures
