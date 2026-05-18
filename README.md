
# Intelligent Geospatial Sampling Quality Control (QC) Validation System

A robust machine learning pipeline designed to automate, optimize, and scale the verification of geospatial plot sampling data across multiple field activities.

## 📌 Project Overview

In large-scale agricultural operations (such as Sugar Group Companies), field inspection data is collected via GPS tracking points across multiple tasks (e.g., *Basal, Brushing, Ecer, Cover, Drop, Harrowing, Keremahan, Pre-Emergence*). Ensuring that sampling points are actually captured **within** the correct boundary lines (polygons) is crucial for data integrity.

This project replaces manual, time-consuming map validation with an automated, high-precision geospatial classification engine. Powered by advanced feature engineering (vector dynamics and distance-to-centroid topologies) and benchmarked using `LazyPredict`, the system deploys a fine-tuned **XGBoost Classifier** achieving **95.15% Accuracy** and a **0.9931 ROC AUC**.

---

## 🛠 Repository Architecture

The repository is split into two primary operational phases:

```text
├── Data Retrieval and Preparation.ipynb   # Phase 1: Database ingestion, cleaning, & target definition
├── QC Point Validation System.ipynb       # Phase 2: UTM projection, feature engineering, benchmarking, & tuning
├── README.md                              # Project documentation
└── requirements.txt                       # Python dependencies

```

---

## 🛰 Project Workflow & Pipeline

```
[ ODA Database ] ➔ [ Data Extraction & Cleaning ] ➔ [ UTM Zone 48S Metric Projection ]
                                                                 ↓
[ Tuned XGBoost Deployment ] 🗂 🧾 [ Hyperparameter Tuning ] 🏓 [ Feature Engineering ]

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

To avoid model bias, `LazyClassifier` evaluated **32 different machine learning architectures** over 14 numeric geospatial features. The top-performing candidates achieved exceptional baseline results:

| Model | Accuracy | Balanced Accuracy | ROC AUC | F1 Score |
| --- | --- | --- | --- | --- |
| **LGBMClassifier** | **0.97** | 0.97 | 0.97 | 0.97 |
| **XGBClassifier** | **0.97** | 0.97 | 0.97 | 0.97 |
| ExtraTreesClassifier | 0.96 | 0.96 | 0.96 | 0.96 |
| RandomForestClassifier | 0.96 | 0.96 | 0.96 | 0.96 |

### Phase 2: Final Fine-Tuning Optimization

While LightGBM and XGBoost scored similarly during the initial run, **XGBoost** was selected due to its structured regularization capabilities, preventing overfitting along ambiguous polygon boundaries.

Following a `GridSearchCV` hyperparameter optimization pass, the model delivered robust generalization capabilities on unseen testing splits:

| Evaluation Metric | Score | Status |
| --- | --- | --- |
| **Accuracy** | **0.9515** | Excellent |
| **F1-Score (Macro)** | **0.9514** | Highly Robust |
| **ROC AUC** | **0.9931** | Superior Class Separation |

> **Operational Insight:** The high ROC AUC (0.9931) proves that the model possesses an elite capacity to distinguish between true spatial drift (invalid data collection) versus standard signal noise.

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

---

## 📈 Operational Roadmap

* [x] Formulate extraction query constraints for production databases.
* [x] Design vector movement feature engineering engine using projected meters.
* [x] Standardize preprocessing routines into an immutable pipeline.
* [ ] Export the final serialized asset via `joblib`/`pickle` to serve API endpoints.
* [ ] Integrate with an analytical platform dashboard (e.g., Tableau or web-based maps) to flag anomalies visually for operations teams.

---

**Developed by:** Muhammad Erico Ricardo
