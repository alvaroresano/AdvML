# Imbalanced Data, Imputation Methods, and Multi-class Classification

This project explores a comprehensive Machine Learning pipeline focused on **Financial Market Prediction** using the NASDAQ-100 and related assets. The core of the research lies in comparing **Data Imputation** techniques and **Imbalance Handling** strategies, followed by an **Explainability** analysis.

The project is structured as a series of three interconnected notebooks, supported by a modular utility library.

## 📂 Project Structure

```text
.
├── notebooks/
│   ├── 01_EDA_and_Target_Creation.ipynb    # Data analysis & labeling
│   ├── 02_Imputation_Methods.ipynb         # Missing data experiments
│   └── 03_Imbalanced_Classification.ipynb  # Modeling & explainability
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
* **Target Definition:** Next-day NASDAQ log returns are categorized into classes based on standard deviations ($\mu \pm 1.5\sigma$):
    * `0: Strong Drop`, `1: Mild Drop`, `2: Neutral`, `3: Mild Rise`, `4: Strong Rise`.
* **Missingness Injection:** To test robustness, we artificially introduce **5% MCAR (Missing Completely At Random)** values into the dataset.
* **Outputs:** Class distribution plots and the corrupted dataset for the next phase.

### 2. Multivariate Imputation Comparison
**Notebook:** `02_Imputation_Methods.ipynb`
* **Objective:** Determine the best strategy to recover missing values in high-correlation financial data.
* **Methods Tested:**
    * **Univariate:** Mean and Median imputation.
    * **Multivariate:** KNN Imputer ($k=5$) and **MICE** (Multivariate Imputation by Chained Equations).
* **Evaluation:** Imputers are evaluated based on the **Downstream F1-Score** of a Random Forest classifier.
* **Key Finding:** MICE preserves the feature covariance structure significantly better than univariate methods.

### 3. Imbalanced Classification & Explainability
**Notebook:** `03_Imbalanced_Classification_&_Explainability.ipynb`
* **Objective:** Tackle the structural imbalance (the "Neutral" class dominates $\approx 52\%$ of the data) and interpret the model.
* **Imbalance Strategies:**
    * Cost-sensitive learning (**Class Weights**).
    * Over-sampling (**SMOTE, ADASYN**).
    * Under-sampling (**RandomUnderSampler**).
    * Combined methods (**SMOTEENN**).
* **Optimization:** Hyperparameter tuning using `GridSearchCV` with `StratifiedKFold` inside an `imblearn` Pipeline to prevent data leakage.
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
