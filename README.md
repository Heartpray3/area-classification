# Area Classification Pipeline

End-to-end machine learning pipeline for multi-class area change classification from satellite imagery-derived geospatial features.

## Task Overview

The objective is to classify each geographical area into one of six classes:

- `0`: Demolition
- `1`: Road
- `2`: Residential
- `3`: Commercial
- `4`: Industrial
- `5`: Mega Projects

Each sample is described by:

- an irregular polygon geometry,
- categorical status values observed at five dates,
- neighborhood urban descriptors,
- neighborhood geographic descriptors.

## Dataset and Scientific Reference

The data format and challenge setup are based on the QFabric dataset:

- **QFabric: Multi-Task Change Detection Dataset**
- Sagar Verma, Akash Panigrahi, Siddharth Gupta
- CVPR Workshops 2021
- Paper: https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.html

BibTeX:

```bibtex
@InProceedings{Verma_2021_CVPR,
  author = {Verma, Sagar and Panigrahi, Akash and Gupta, Siddharth},
  title = {QFabric: Multi-Task Change Detection Dataset},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {June},
  year = {2021},
  pages = {1052-1061}
}
```

## Repository Structure

- `preprocess.py` - preprocessing entry point (recommended)
- `preprocess_features.py` - core feature engineering implementation
- `train.py` - model training, cross-validation and test prediction
- `requirements.txt` - Python dependencies
- `reports/` - report/reference documents

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input Data

Place the following files in the project root:

- `train.geojson`
- `test.geojson`

## 1) Feature Generation

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

Generated files:

- `train_features.parquet`
- `test_features.parquet`

## 2) Training and Prediction

```bash
python3 train.py
```

Generated file:

- `submission.csv`

The current training script includes cross-validation reporting (accuracy, balanced accuracy, F1 weighted/macro, per-class F1) and then trains on full training data before predicting on test data.

## Validation Performance

On the validation set, the model achieved **88% weighted F1-score**.

## Evaluation Protocol

The target metric is the **Mean F1-Score** for multi-class classification.

For leaderboard submission, `submission.csv` must follow:

```text
Id,change_type
0,2
1,3
...
```

## Feature Engineering Summary

The preprocessing pipeline builds:

- geometry features (area, perimeter, compactness, rotated-rectangle shape features, etc.),
- temporal progression features (durations, transitions, status-change frequency),
- one-hot and derived urban/geography features,
- timeline image-derived features across dates.

## License

This repository is distributed under the MIT License (see `LICENSE`).
