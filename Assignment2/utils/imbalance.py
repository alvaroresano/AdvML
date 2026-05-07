"""
utils/imbalance.py
==================
Imbalanced data handling strategies covered in Unit 3:
  - Class weights (balanced)
  - Over-sampling: RandomOverSampler, SMOTE, ADASYN
  - Under-sampling: RandomUnderSampler
  - Combined: SMOTEENN
 
Each strategy is wrapped in an sklearn Pipeline together with a
StandardScaler and a RandomForestClassifier so experiments are
directly comparable.
"""
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    confusion_matrix,
)
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
 
 
# ---------------------------------------------------------------------------
# Strategy catalogue
# ---------------------------------------------------------------------------
 
def safe_k_neighbors(y, default: int = 5) -> int:
    """
    SMOTE/ADASYN need at least k+1 samples in the smallest class within every
    CV fold. Returns min(default, n_min - 1) with a floor of 1 so the algorithm
    never crashes on tiny minority classes.
    """
    counts = pd.Series(y).value_counts()
    n_min = int(counts.min())
    return max(1, min(default, n_min - 1))


def get_strategies(random_state: int = 42, y_train=None) -> dict:
    """
    Return a dict mapping strategy name → imblearn/sklearn Pipeline.

    All pipelines follow the same structure:
      [optional sampler] → StandardScaler → RandomForestClassifier

    If `y_train` is provided, k_neighbors for SMOTE/ADASYN is shrunk to
    a safe value when the minority class is small (avoids crashes inside CV).
    """
    rf_params = dict(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )
    k = safe_k_neighbors(y_train) if y_train is not None else 5

    strategies = {
        # ------------------------------------------------------------------ #
        # Baseline (no handling)                                              #
        # ------------------------------------------------------------------ #
        "Baseline (no handling)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**rf_params)),
        ]),

        # ------------------------------------------------------------------ #
        # Class weights only (no resampling)                                  #
        # ------------------------------------------------------------------ #
        "Class Weights (balanced)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**rf_params, class_weight="balanced")),
        ]),

        # ------------------------------------------------------------------ #
        # Over-sampling                                                       #
        # ------------------------------------------------------------------ #
        "RandomOverSampler": ImbPipeline([
            ("sampler", RandomOverSampler(random_state=random_state)),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**rf_params)),
        ]),
        "SMOTE": ImbPipeline([
            ("sampler", SMOTE(random_state=random_state, k_neighbors=k)),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**rf_params)),
        ]),
        "ADASYN": ImbPipeline([
            ("sampler", ADASYN(random_state=random_state, n_neighbors=k)),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**rf_params)),
        ]),

        # ------------------------------------------------------------------ #
        # Under-sampling                                                      #
        # ------------------------------------------------------------------ #
        "RandomUnderSampler": ImbPipeline([
            ("sampler", RandomUnderSampler(random_state=random_state)),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**rf_params)),
        ]),

        # ------------------------------------------------------------------ #
        # Combined (SMOTE + ENN cleaning)                                    #
        # ------------------------------------------------------------------ #
        "SMOTEENN": ImbPipeline([
            ("sampler", SMOTEENN(random_state=random_state,
                                 smote=SMOTE(random_state=random_state, k_neighbors=k))),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(**rf_params)),
        ]),
    }
    return strategies
 
 
# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
 
def evaluate_strategy(
    name: str,
    pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: dict | None = None,
) -> dict:
    """
    Fit a strategy pipeline and evaluate on the test set.
    
    Returns a rich result dict for downstream analysis.
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
 
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
 
    target_names = None
    if class_names:
        labels = sorted(class_names.keys())
        target_names = [class_names[k] for k in labels]
 
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0,
    )
 
    return {
        "strategy": name,
        "pipeline": pipeline,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "y_pred": y_pred,
    }
 
 
def compare_strategies(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: dict | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all strategies and return a summary DataFrame and full results dict.
    """
    strategies = get_strategies(random_state=random_state)
    all_results = {}
    rows = []
 
    for name, pipeline in strategies.items():
        print(f"  Running: {name}...", end=" ", flush=True)
        r = evaluate_strategy(name, pipeline, X_train, y_train, X_test, y_test, class_names)
        all_results[name] = r
        row = {
            "Strategy": name,
            "Macro F1": round(r["macro_f1"], 4),
            "Weighted F1": round(r["weighted_f1"], 4),
        }
        for i, f1 in enumerate(r["per_class_f1"]):
            label = class_names.get(i, str(i)) if class_names else str(i)
            row[f"F1 [{label}]"] = round(f1, 4)
        rows.append(row)
        print(f"Macro F1={r['macro_f1']:.4f}")
 
    summary_df = pd.DataFrame(rows).set_index("Strategy")
    return summary_df, all_results