# Imbalanced Data, Imputation Methods, and Multi-class Classification

This project explores a comprehensive Machine Learning pipeline focused on **Financial Market Prediction** using the NASDAQ-100 and related assets. The core of the research lies in comparing **Data Imputation** techniques and **Imbalance Handling** strategies, followed by an **Explainability** analysis.

The project is structured as a series of three interconnected notebooks, supported by a modular utility library.

## 📂 Project Structure

```text
.
├── notebooks/
│   ├── 01_EDA_and_Target_Creation.ipynb                       # Data analysis & labeling
│   ├── 02_Imputation_Methods.ipynb                            # Missing data experiments
│   └── 03_Imbalanced_Classification_&_Explainability.ipynb    # Modeling & explainability
├── utils/                                  # Modular Python scripts
│   ├── data_loader.py                      # Preprocessing & feature engineering
│   ├── imputation.py                       # Imputation experiment helpers
│   ├── imbalance.py                        # Resampling strategies (SMOTE, etc.)
│   └── evaluation.py                       # Visualization & metrics
├── data/                                   # Parquet datasets (Clean & Missing)
└── outputs/                                # Generated figures, reports & SHAP plots
```

---

## 🛠️ Workflow Details

### 1. Exploratory Data Analysis & Target Engineering
**Notebook:** `01_EDA_and_Target_Creation.ipynb`
* **Objective:** Transform the regression problem (predicting returns) into a **5-class ordinal classification** task.
* **Target Definition:** Next-day NASDAQ log returns are categorised using $\mu \pm 0.5\sigma$ and $\mu \pm 1.5\sigma$ as cut-points:
    * `0: Strong Drop`, `1: Mild Drop`, `2: Neutral`, `3: Mild Rise`, `4: Strong Rise`.
    * In notebook 01 the thresholds are computed on the full series **for descriptive purposes only**; the predictive notebooks (02, 03) recompute them on the training split alone.
* **Missingness Injection:** **5% MCAR** values are introduced into 8 technical-indicator columns to enable a controlled imputation experiment.
* **Outputs:** Class distribution plots and the corrupted dataset for the next phase.

### 2. Multivariate Imputation Comparison
**Notebook:** `02_Imputation_Methods.ipynb`
* **Objective:** Determine the best strategy to recover missing values in high-correlation financial data.
* **Split:** **Chronological** (last 20% by date), not stratified-random — the data is autocorrelated and shuffling leaks temporal context across the boundary.
* **Methods Tested:**
    * **Univariate:** Mean and Median imputation.
    * **Multivariate:** KNN Imputer ($k=5$) and **MICE** (Multivariate Imputation by Chained Equations).
* **Evaluation:** Two complementary criteria —
    1. **Reconstruction quality** (MAE / RMSE per corrupted column against the held-out clean reference).
    2. **Downstream F1-score** of an unweighted Random Forest classifier.
* **Key Finding:** MICE preserves the feature covariance structure significantly better than univariate methods, especially on columns highly correlated with the rest of the feature space.

### 3. Imbalanced Classification & Explainability
**Notebook:** `03_Imbalanced_Classification_&_Explainability.ipynb`
* **Objective:** Tackle the structural imbalance (the "Neutral" class dominates $\approx 52\%$ of the data) and interpret the model.
* **Floor benchmarks:** `DummyClassifier(most_frequent | stratified)` plus a plain Random Forest, providing the reference against which every strategy is judged (slide 7 of Unit 3).
* **Imbalance Strategies:**
    * Cost-sensitive learning (**Class Weights**).
    * Over-sampling (**SMOTE, ADASYN**, with a defensive `k_neighbors`).
    * Under-sampling (**RandomUnderSampler**).
    * Combined methods (**SMOTEENN**).
* **Metrics:** Macro F1 (primary), balanced accuracy and Cohen's κ.
* **Optimisation:** Hyperparameter tuning using `GridSearchCV` with **`TimeSeriesSplit`** (5 folds) inside an `imblearn` Pipeline. `class_weight` is deliberately removed from the grid because SMOTEENN already balances the classes — re-weighting on top would over-penalise the resampled majority. The tuned pipeline is persisted to `outputs/models/best_pipeline.joblib`.
* **Explainability:** Implementation of **SHAP (SHapley Additive exPlanations)** to identify which technical indicators (RSI, Volatility, Lags) trigger extreme market movement predictions.

---

## 📊 Outputs & Reports
The notebooks automatically populate the `outputs/` folder with:
* **Visualizations:** Heatmaps of missing data, confusion matrices for every resampling strategy, and F1-score comparison bar charts.
* **Reports:** Detailed classification reports and performance summaries in CSV/Markdown format.
* **SHAP Plots:** Summary and Beehive plots explaining the impact of financial features on specific market classes.

---

## 💻 Technical Stack
* **Core:** `Python 3.12+`
* **Data Handling:** `Pandas`, `Numpy`
* **Machine Learning:** `Scikit-Learn`, `Imbalanced-Learn` (imblearn)
* **Explainability:** `SHAP`
* **Visualization:** `Matplotlib`, `Seaborn`

## 🚀 How to Run
1. **Prepare Data:** Ensure the raw financial CSV is available in the `data/` folder.
2. **Modular Utils:** The `utils/` folder must be in your Python path (same level as notebooks).
3. **Sequence:** Run the notebooks in order (`01` → `02` → `03`) to ensure all intermediate `.parquet` files and `outputs/` are generated correctly.
