import json
from enum import Enum
from typing import List, Dict, Any

class EvalStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    BLOCKED = "BLOCKED"
    INCONCLUSIVE = "INCONCLUSIVE"
    ERROR = "ERROR"

class FixtureManager:
    """
    Handles loading and validation of versioned evaluation fixtures.
    """



    def __init__(self, fixture_dir: str):
        self.fixture_dir = fixture_dir

    def load_raw_mock(self, file_id: str) -> Dict[str, Any]:
        path = f"{self.fixture_dir}/raw/studies/{file_id}.json"
        with open(path, 'r') as f:
            return json.load(f)

    def load_state_mock(self, state_id: str) -> Dict[str, Any]:
        path = f"{self.fixture_dir}/states/{state_id}.json"
        with open(path, 'r') as f:
            return json.load(f)

    def load_reference_output(self, case_id: str) -> Dict[str, Any]:
        path = f"{self.fixture_dir}/references/{case_id}.json"
        with open(path, 'r') as f:
            return json.load(f)

    def validate_fixture(self, data: Any, schema: Any) -> bool:
        # Simple check for dictionary type as basic validation
        return isinstance(data, dict) if schema == "dict" else True
