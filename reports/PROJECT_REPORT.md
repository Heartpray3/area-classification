# Project Report

## Objective

Build a robust area change classification pipeline from geospatial data.

## Pipeline Overview

1. `preprocess.py`
   - Loads `train.geojson` and `test.geojson`
   - Computes geometry, timeline, temporal transition, and neighborhood features
   - Exports:
     - `train_features.parquet`
     - `test_features.parquet`

2. `train.py`
   - Loads engineered parquet features
   - Trains an XGBoost classifier with stratified cross-validation
   - Generates:
     - `submission.csv`

## Feature Groups Used

- Geometry features (area, perimeter, compactness, etc.)
- Urban and geography one-hot features
- Urban/geography derived features
- Temporal and transition features
- Timeline image-derived features
- Status one-hot flags per date

## Repository Notes

- Main scripts:
  - `preprocess.py`
  - `train.py`
- Core feature engineering module:
  - `preprocess_features.py`
