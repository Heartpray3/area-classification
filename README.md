# Area Classification Pipeline

End-to-end pipeline for area change classification:
- `preprocess.py`: builds engineered features from GeoJSON
- `train.py`: trains the model with CV and generates `submission.csv`

## Project Structure

- `preprocess.py` - main preprocessing entry point (recommended script)
- `preprocess_features.py` - core feature engineering implementation
- `train.py` - training + cross-validation + submission generation
- `reports/` - project reports and supporting documents
- `requirements.txt` - Python dependencies

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Files Expected

Place these files in the project root:
- `train.geojson`
- `test.geojson`

## 1) Generate Features

```bash
python3 preprocess.py
```

Optional custom paths:

```bash
python3 preprocess.py \
  --train train.geojson \
  --test test.geojson \
  --train-out train_features.parquet \
  --test-out test_features.parquet
```

## 2) Train + Create Submission

```bash
python3 train.py
```

Outputs:
- Console CV metrics (accuracy, balanced accuracy, F1 weighted/macro, per-class F1)
- `submission.csv`

## Notes

- The preprocessing includes:
  - geometry features (area, perimeter, compactness, etc.)
  - temporal/status transitions and durations
  - urban/geography one-hot and derived features
  - timeline image-derived features
- `train.py` consumes all engineered feature groups and also includes a numeric fallback to avoid missing newly added numeric features.

## GitHub Hygiene

Recommended files to avoid committing:
- `.venv/`
- `__pycache__/`
- generated artifacts (`*.parquet`, `submission.csv`)
