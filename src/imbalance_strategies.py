"""
Class Imbalance Handling and Benchmarking Module.

This module implements and evaluates different strategies for handling
extreme class imbalance (0.17% fraud prevalence):
1. SMOTE (Synthetic Minority Over-sampling Technique)
2. ADASYN (Adaptive Synthetic Sampling)
3. scale_pos_weight (Cost-sensitive loss weighting in LightGBM)

Evaluates strategies using PR-AUC as the primary metric, since standard
ROC-AUC is overoptimistic in high-imbalance regimes.
"""

from typing import Any, Dict, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 500,
    "random_state": 42,
    "verbose": -1,
}


def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate negative-to-positive class ratio for cost-sensitive training.

    Args:
        y_train: Binary training labels (0 = Legit, 1 = Fraud).

    Returns:
        float ratio of negative count to positive count.
    """
    neg_count = float((y_train == 0).sum())
    pos_count = float((y_train == 1).sum())
    if pos_count == 0:
        return 1.0
    return neg_count / pos_count


def evaluate_resampling_strategy(
    strategy_name: str,
    X_train_resampled: np.ndarray,
    y_train_resampled: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_params: Optional[Dict[str, Any]] = None,
    early_stopping_rounds: int = 50,
) -> Dict[str, Any]:
    """
    Train LightGBM on resampled data and evaluate on unmodified test data.

    Args:
        strategy_name: Identifier for the strategy (e.g., 'SMOTE', 'ADASYN').
        X_train_resampled: Resampled or weighted feature array.
        y_train_resampled: Resampled or weighted target labels.
        X_test: Original test feature set.
        y_test: Original test target labels.
        model_params: Hyperparameter dictionary for LightGBM.
        early_stopping_rounds: Number of early stopping rounds.

    Returns:
        Dictionary of performance metrics and trained model.
    """
    if model_params is None:
        model_params = dict(DEFAULT_LGBM_PARAMS)

    model = lgb.LGBMClassifier(**model_params)
    model.fit(
        X_train_resampled,
        y_train_resampled,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        "Strategy": strategy_name,
        "PR-AUC": round(float(pr_auc), 4),
        "ROC-AUC": round(float(roc_auc), 4),
        "F1@0.5": round(float(f1), 4),
        "Model": model,
        "Probabilities": y_prob,
    }


def compare_imbalance_strategies(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    base_params: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Benchmark SMOTE, ADASYN, and scale_pos_weight strategies.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Test features.
        y_test: Test labels.
        base_params: Base parameters for LightGBM.
        random_state: Random seed for samplers.

    Returns:
        Tuple of (comparison_dataframe, dictionary_of_detailed_results).
    """
    if base_params is None:
        base_params = dict(DEFAULT_LGBM_PARAMS)

    results_detail: Dict[str, Any] = {}
    summary_rows = []

    # 1. SMOTE
    print("Evaluating Strategy 1/3: SMOTE...")
    smote = SMOTE(random_state=random_state, k_neighbors=5)
    X_sm, y_sm = smote.fit_resample(X_train, y_train)
    smote_out = evaluate_resampling_strategy("SMOTE", X_sm, y_sm, X_test, y_test, base_params)
    results_detail["SMOTE"] = smote_out
    summary_rows.append({
        "Strategy": "SMOTE",
        "PR-AUC": smote_out["PR-AUC"],
        "ROC-AUC": smote_out["ROC-AUC"],
        "F1@0.5": smote_out["F1@0.5"],
    })

    # 2. ADASYN
    print("Evaluating Strategy 2/3: ADASYN...")
    adasyn = ADASYN(random_state=random_state, n_neighbors=5)
    X_ad, y_ad = adasyn.fit_resample(X_train, y_train)
    adasyn_out = evaluate_resampling_strategy("ADASYN", X_ad, y_ad, X_test, y_test, base_params)
    results_detail["ADASYN"] = adasyn_out
    summary_rows.append({
        "Strategy": "ADASYN",
        "PR-AUC": adasyn_out["PR-AUC"],
        "ROC-AUC": adasyn_out["ROC-AUC"],
        "F1@0.5": adasyn_out["F1@0.5"],
    })

    # 3. scale_pos_weight (Cost-sensitive)
    print("Evaluating Strategy 3/3: scale_pos_weight...")
    spw = compute_scale_pos_weight(y_train)
    cost_params = {**base_params, "scale_pos_weight": spw}
    cost_out = evaluate_resampling_strategy("scale_pos_weight", X_train, y_train, X_test, y_test, cost_params)
    results_detail["scale_pos_weight"] = cost_out
    summary_rows.append({
        "Strategy": "scale_pos_weight",
        "PR-AUC": cost_out["PR-AUC"],
        "ROC-AUC": cost_out["ROC-AUC"],
        "F1@0.5": cost_out["F1@0.5"],
    })

    comparison_df = pd.DataFrame(summary_rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    return comparison_df, results_detail
