"""
utils/imputation.py
===================
Imputation methods comparison:
  - SimpleImputer (mean, median) → Univariate
  - KNNImputer                  → Multivariate
  - IterativeImputer (MICE)     → Multivariate
  
All imputers are always FITTED on the training set only and then
applied to transform the test set.  This prevents data leakage.
"""
 
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
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
 
    Returns a dict with macro F1 and per-class F1 scores.
    """
    # Fit ONLY on training data (critical: avoids data leakage)
    X_train_imp = imputer.fit_transform(X_train_missing)
    X_test_imp = imputer.transform(X_test_missing)
 
    clf = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
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