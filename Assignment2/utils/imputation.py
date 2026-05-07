"""
utils/imputation.py
===================
Imputation methods comparison:
  - SimpleImputer (mean, median) → Univariate
  - KNNImputer                  → Multivariate
  - IterativeImputer (MICE)     → Multivariate

All imputers are always FITTED on the training set only and then
applied to transform the test set.  This prevents data leakage.

Two evaluation criteria are provided:
  1. Reconstruction quality (MAE / RMSE per corrupted column) using a
     held-out clean reference — measures how faithfully the imputer
     recovers the missing entries.
  2. Downstream classification F1 — measures whether the imputation
     choice changes a real predictive task.
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
 
 
# ---------------------------------------------------------------------------
# Imputer catalogue
# ---------------------------------------------------------------------------
 
def get_imputers() -> dict:
    """
    Return a dictionary of named imputer objects following the
    sklearn Transformer API (fit / transform).
    """
    return {
        "Mean (univariate)": SimpleImputer(strategy="mean"),
        "Median (univariate)": SimpleImputer(strategy="median"),
        "KNN (k=5)": KNNImputer(n_neighbors=5),
        "MICE / IterativeImputer": IterativeImputer(
            max_iter=10, random_state=42, tol=1e-3
        ),
    }
 
 
# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------
 
def evaluate_imputer(
    imputer_name: str,
    imputer,
    X_train_missing: np.ndarray,
    y_train: np.ndarray,
    X_test_missing: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict:
    """
    Fit the imputer on training data, impute both train and test,
    then train a baseline RandomForest and evaluate on test.

    The downstream classifier is intentionally unweighted here so the
    F1 differences across imputers reflect imputation quality rather
    than reweighting effects (imbalance handling is studied separately
    in notebook 03).

    Returns a dict with macro F1 and per-class F1 scores.
    """
    # Fit ONLY on training data (critical: avoids data leakage)
    X_train_imp = imputer.fit_transform(X_train_missing)
    X_test_imp = imputer.transform(X_test_missing)

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X_train_imp, y_train)
    y_pred = clf.predict(X_test_imp)

    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    return {
        "imputer": imputer_name,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "y_pred": y_pred,
        "clf": clf,
        "imputer_obj": imputer,
    }


# ---------------------------------------------------------------------------
# Reconstruction-quality evaluation (requires the clean reference)
# ---------------------------------------------------------------------------

def reconstruction_quality(
    imputer,
    X_train_missing: pd.DataFrame,
    X_test_missing: pd.DataFrame,
    X_test_clean: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    """
    Fit the imputer on the corrupted training set, transform the corrupted
    test set, and compare imputed values against the ground-truth ``X_test_clean``
    on the entries that were artificially removed.

    Returns a DataFrame with MAE and RMSE per target column.
    """
    cols_all = list(X_train_missing.columns)
    X_test_imp = imputer.fit(X_train_missing.values).transform(X_test_missing.values)

    rows = []
    for col in target_cols:
        if col not in cols_all:
            continue
        idx = cols_all.index(col)
        was_missing = X_test_missing[col].isnull().values
        if was_missing.sum() == 0:
            continue
        true_vals = X_test_clean[col].values[was_missing]
        pred_vals = X_test_imp[was_missing, idx]
        mae = mean_absolute_error(true_vals, pred_vals)
        rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
        rows.append({
            "column": col,
            "n_missing": int(was_missing.sum()),
            "MAE": round(mae, 6),
            "RMSE": round(rmse, 6),
        })
    return pd.DataFrame(rows)


def compare_reconstruction(
    X_train_missing: pd.DataFrame,
    X_test_missing: pd.DataFrame,
    X_test_clean: pd.DataFrame,
    target_cols: list[str],
) -> pd.DataFrame:
    """
    Run every imputer in ``get_imputers()`` and stack the per-column
    MAE / RMSE into a single comparison DataFrame.
    """
    frames = []
    for name, imputer in get_imputers().items():
        q = reconstruction_quality(
            imputer, X_train_missing, X_test_missing, X_test_clean, target_cols
        )
        q.insert(0, "Imputer", name)
        frames.append(q)
    return pd.concat(frames, ignore_index=True)
 
 
def compare_imputers(
    X_train_missing: np.ndarray,
    y_train: np.ndarray,
    X_test_missing: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Run all imputers and return a comparison DataFrame.
    """
    results = []
    for name, imputer in get_imputers().items():
        r = evaluate_imputer(
            name, imputer,
            X_train_missing, y_train,
            X_test_missing, y_test,
        )
        row = {"Imputer": name, "Macro F1": round(r["macro_f1"], 4)}
        for i, f1 in enumerate(r["per_class_f1"]):
            row[f"F1 Class {i}"] = round(f1, 4)
        results.append(row)
 
    return pd.DataFrame(results).set_index("Imputer")