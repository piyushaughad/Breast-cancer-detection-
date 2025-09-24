# Breast Cancer Prediction - Predictive Modelling for Diagnosis

**One-line:** Predict and compare machine learning models (SVM, Decision Tree, Random Forest) to diagnose breast cancer from the Breast Cancer Wisconsin (Diagnostic) dataset.

---

## Quick answers - What / How / Why

### What?
A supervised ML project that predicts whether a tumor is **malignant** or **benign** using features extracted from digitized breast mass images. Models used: **Support Vector Machine (SVM)**, **Decision Tree**, and **Random Forest**. Dataset and experimental results come from our project report.

### How?
Step-by-step:  
1. Obtain the dataset (UCI Breast Cancer Wisconsin (Diagnostic)).  
2. Preprocess (drop ID, encode diagnosis, handle outliers, scale features, remove highly correlated features).  
3. Split data (70% train / 30% test).  
4. Train SVM (RBF kernel, tuned `C` and `gamma`), Decision Tree (with/without pruning), and Random Forest (ensemble with tuned `n_estimators`, `max_features`, etc.).  
5. Evaluate with accuracy, recall, precision, F1 and confusion matrix.  
6. Save the best model for deployment. Full details and metrics are in the project report. :contentReference[oaicite:2]{index=2}

### Why?
Early and accurate breast cancer diagnosis increases treatment success and reduces mortality. Machine learning can help clinicians by providing fast, objective, reproducible predictions that support diagnostic decisions. This comparative study identifies which algorithms perform best on this dataset and how to prepare the data for robust results. :contentReference[oaicite:3]{index=3}

---

## Project overview
This repo implements and compares machine learning models to predict breast cancer diagnosis (malignant vs benign) from the Breast Cancer Wisconsin (Diagnostic) dataset. The aim is to explore preprocessing choices, feature selection, hyperparameter tuning, and model evaluation to identify the most reliable classifier for this dataset. :contentReference[oaicite:4]{index=4}

---

## Dataset
- **Source:** Breast Cancer Wisconsin (Diagnostic) dataset (UCI ML repository).  
- **Records:** 569 samples, 32 attributes (ID, diagnosis, 30 numeric features capturing mean / se / worst of cell nucleus measurements).  
- **Target:** `diagnosis` (M = malignant, B = benign). :contentReference[oaicite:5]{index=5}

> Note: If you include the raw CSV in the repo, add it as `/data/breast_cancer_wisconsin.csv`. Otherwise add a short script to download it from the UCI link.

---

## Preprocessing & EDA (what we did)
Typical steps (as implemented in the notebook / scripts):

1. **Load CSV** and inspect rows/columns.  
2. **Drop non-informative columns** (e.g., `id`, `Unnamed: 32`).  
3. **Encode target**: `M` → `1`, `B` → `0`.  
4. **Missing values**: check + impute or drop (dataset typically has no missing values).  
5. **Outlier handling**: identify with boxplots and cap extreme values (IQR method).  
6. **Feature correlation**: compute correlation matrix and remove one column from pairs with very high correlation (to reduce multicollinearity).  
7. **Scaling**: standardize features (StandardScaler) before SVM.  
8. **Train/test split**: 70% train / 30% test.  
9. **Visual EDA**: histograms, boxplots, heatmap of correlations. :contentReference[oaicite:6]{index=6}

---

## Modeling approach (how)
- **Support Vector Machine (SVM)**  
  - Kernel: RBF (radial basis) proved effective.  
  - Hyperparameters tuned: `C` (regularization), `gamma`.  
  - Example: `C=10` and RBF produced strong results in the study. :contentReference[oaicite:7]{index=7}

- **Decision Tree**  
  - Splitting criterion: Gini or entropy.  
  - Pruning: perform pre-pruning (max_depth, min_samples_leaf) or post-pruning (ccp_alpha) to avoid overfitting.

- **Random Forest**  
  - Ensemble of decision trees with bootstrap sampling and random feature selection per split.  
  - Common hyperparameters: `n_estimators`, `max_features`, `min_samples_split`, `min_samples_leaf`.

- **Evaluation**: accuracy, recall (sensitivity), precision, F1 score, and confusion matrix. Cross-validation or grid search used for tuning.

---

## Results (high level)
Observed from the project experiments (reported in the group report):

| Model          | Accuracy (%) | Recall (%) | Precision (%) | F1 (%) |
| -------------- | ------------:| ----------:| -------------:| ------:|
| SVM (RBF, C=10)| 98           | 97         | 98            | 98     |
| Random Forest  | 95           | 94         | 94            | 94     |
| Decision Tree  | 92           | 90         | 89            | 90     |

---

## How to run (step-by-step)
Follow these steps to reproduce the experiments:

1. **Clone the repo**
   ```bash
   git clone https://github.com/<your-username>/breast-cancer-prediction.git
   cd breast-cancer-prediction
