# Intelligent Geospatial Sampling Quality Control (QC) Validation System

A robust machine learning pipeline designed to automate, optimize, and scale the verification of geospatial plot sampling data across multiple field activities.

## 📌 Project Overview

In large-scale agricultural operations (such as Sugar Group Companies), field inspection data is collected via GPS tracking points across multiple tasks (e.g., *Basal, Brushing, Ecer, Cover, Drop, Harrowing, Keremahan, Pre-Emergence*). Ensuring that sampling points are actually captured **within** the correct boundary lines (polygons) is crucial for data integrity.

This project replaces manual, time-consuming map validation with an automated, high-precision geospatial classification engine. Powered by advanced feature engineering (vector dynamics and distance-to-centroid topologies) and benchmarked using `LazyPredict`, the system deploys an **XGBoost Classifier** (baseline/default configuration) achieving **96.94% Accuracy** and a **0.9794 ROC AUC** on unseen test data.

---

## 🛠 Repository Architecture

The repository is split into two primary operational phases:

```text
├── Data Retrieval and Preparation.ipynb       # Phase 1: Database ingestion, cleaning, & target definition
├── QC_Point_Validation_System_V5.ipynb        # Phase 2: UTM projection, feature engineering, benchmarking, & tuning
├── README.md                                  # Project documentation
└── requirements.txt                           # Python dependencies

```

---

## 🛰 Project Workflow & Pipeline

```
[ ODA Database ] ➔ [ Data Extraction & Cleaning ] ➔ [ UTM Zone 48S Metric Projection ]
                                                                 ↓
[ Baseline XGBoost Deployment ] 🗂 🧾 [ Hyperparameter Tuning (evaluated) ] 🏓 [ Feature Engineering ]

```

### 1. Data Ingestion & Preparation (`Data Retrieval and Preparation.ipynb`)

Data is pulled directly from the **ODA Production Database** targeting the **2025 Crop Season**. The ingestion pipeline performs the following tasks:

* **Consolidation**: Implements an optimized `UNION ALL` SQL structure to aggregate coordinates and `WKT (Well-Known Text)` shape data from diverse inspection tables.
* **Rigorous Filtering**:
  * Excludes zero-accuracy GPS readings (`AKURASI != 0`).
  * Drops dead/null coordinate sets (`0.0, 0.0`).
  * Ensures structural integrity by filtering out empty spatial polygons (`SHAPE IS NOT NULL`).
* **Ground Truth Building**: Prepares data rows for sequential spatial matching against manual baseline checks.

### 2. Advanced Geospatial Feature Engineering

Standard WGS84 degree coordinates (latitude/longitude) distort physical distances. To calculate highly accurate distance and direction metrics, the pipeline transforms the world into meters:

* **Metric Reprojection**: Converts coordinate reference systems from `EPSG:4326` to **UTM Zone 48S (EPSG:32748)**.
* **Centroid Proximity Modeling**: Calculates the polygon's exact center point (*centroid*) and determines:
  * `in_i`: A binary marker tracking if individual point *i* falls within the boundary.
  * `dist_ci`: Point-to-centroid linear distance in meters.
  * `angle_i`: Relative directional angle using trigonometric `arctan2`.
* **Kinematic Vector Dynamics**: Calculates `move_angle_12` and `move_angle_23` to decode the sequential direction and path diagonality of inspectors across the field.
* **Global Integrity Indicator (`all_inside`)**: A strict evaluation flag validating that all tracking points share absolute spatial containment.

---

## 📊 Model Evaluation & Benchmarking

### Phase 1: Rapid Model Exploration (`LazyPredict`)

To avoid model bias, `LazyClassifier` evaluated **32 candidate architectures** over 14 numeric geospatial features. The top-performing candidates achieved strong, closely-matched baseline results:

| Model | Accuracy | Balanced Accuracy | ROC AUC | F1 Score |
| --- | --- | --- | --- | --- |
| **RandomForestClassifier** | **0.96** | 0.95 | 0.95 | 0.96 |
| **XGBClassifier** | **0.96** | 0.95 | 0.95 | 0.96 |
| **LGBMClassifier** | **0.96** | 0.95 | 0.95 | 0.96 |
| ExtraTreesClassifier | 0.96 | 0.94 | 0.94 | 0.95 |

`XGBClassifier` was carried forward: it matched the top-scoring models above while training faster than `LGBMClassifier` and offering more structured regularization control than `RandomForestClassifier`.

### Phase 2: Hyperparameter Tuning — the Baseline Won

`GridSearchCV` was run to search for a better configuration than XGBoost's defaults. It found a combination with a higher **internal** cross-validation score, but on the standalone Test set the **baseline (default) configuration outperformed the tuned one** on accuracy, precision, recall, and F1 — the tuned model only improved ROC AUC slightly. Because of this, the **baseline configuration is what's deployed**, not the tuned one.

| Evaluation Metric (Test Set) | Baseline (Deployed) | Tuned (GridSearchCV) |
| --- | --- | --- |
| **Accuracy** | **0.9694** | 0.9617 |
| **F1-Score (Macro)** | **0.9640** | 0.9549 |
| **ROC AUC** | 0.9794 | **0.9803** |

> **Operational Insight:** The baseline model's strong ROC AUC (0.9794) shows it reliably separates true spatial drift (invalid data collection) from ordinary GPS/signal noise — and it does so with half the tree count of the tuned alternative, at lower inference cost.

---

## 💻 Getting Started & Installation

### Prerequisites

Make sure you have a Python environment setup (v3.9+ recommended). Geospatial libraries require proper handling of underlying C dependencies (`shapely`, `fiona`, `pyproj`).

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/geospatial-qc-validation.git
cd geospatial-qc-validation

```

2. Install dependencies:
```bash
pip install -r requirements.txt

```

### Core Stack

* **Geospatial Processing**: `Geopandas`, `Shapely`, `pyproj`
* **Machine Learning Pipeline**: `XGBoost`, `Scikit-Learn`, `LazyPredict`
* **Data & Analytics**: `Pandas`, `Numpy`, `Phik` (Advanced Correlation Analysis)
* **Deployment**: `Streamlit` — see the [GeoValid QC Validator app](https://huggingface.co/spaces/EricoR/SGC_Petak_QC_Validator) for the live inference dashboard

---

## 📈 Operational Roadmap

* [x] Formulate extraction query constraints for production databases.
* [x] Design vector movement feature engineering engine using projected meters.
* [x] Standardize preprocessing routines into an immutable pipeline.
* [x] Export the final serialized asset via `pickle` for API/app integration.
* [x] Integrate with a web-based dashboard (Streamlit) to flag anomalies visually for operations teams.
* [ ] Boundary-case error analysis: manually review misclassified points sitting on/near polygon edges.
* [ ] Re-evaluate hyperparameter tuning with the objective function set to Class 1 Recall and Stratified Repeated K-Fold CV, to see if it can beat the baseline.

---

**Developed by:** Muhammad Erico Ricardo
