# Offline Evaluation Dataset

This dataset contains curated test cases for the MOSAIC research integrity system.

## Structure
Each line is a JSON object containing:
- `task`: The natural language analysis request.
- `expected_signal`: The type of signal the system should ideally detect.
- `nct_id`: The target study ID.
- `expected_confidence`: The minimum confidence expected for a high-quality signal.

## Usage
Run the evaluation script:
```bash
python scripts/evaluate.py
```
