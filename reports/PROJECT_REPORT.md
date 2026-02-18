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

## Validation Performance

On the validation set, we obtained **88% weighted F1-score**.

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

## References

- Verma, S., Panigrahi, A., Gupta, S. (2021). **QFabric: Multi-Task Change Detection Dataset**.  
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 1052-1061.  
  Paper: https://openaccess.thecvf.com/content/CVPR2021W/EarthVision/html/Verma_QFabric_Multi-Task_Change_Detection_Dataset_CVPRW_2021_paper.html

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
