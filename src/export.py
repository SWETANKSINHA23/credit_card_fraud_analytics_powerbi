"""
Dashboard Artifact Export and Reporting Module.

Exports flat, dashboard-ready CSV datasets consumed by Power BI and Tableau:
1. outputs/scored_transactions.csv: Transaction-level predictions, amounts, merchant categories, and labels.
2. outputs/threshold_metrics.csv: Full cost and statistical metric sweep across decision thresholds.
3. outputs/pr_curve_points.csv: High-resolution Precision-Recall curve coordinates.
4. outputs/feature_importance.csv: SHAP importance scores per feature.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


def export_scored_transactions(
    engineered_df: pd.DataFrame,
    test_indices: pd.Index,
    y_test: pd.Series,
    fraud_probabilities: np.ndarray,
    operating_threshold: float,
    output_path: Union[str, Path] = "outputs/scored_transactions.csv",
) -> pd.DataFrame:
    """
    Format and export test set transaction predictions with metadata.

    Args:
        engineered_df: Original DataFrame containing 'Amount', 'hour_of_day', and 'merchant_category'.
        test_indices: Row indices corresponding to the test partition.
        y_test: Actual test ground truth labels.
        fraud_probabilities: Model-predicted positive class probabilities.
        operating_threshold: Decision threshold for binary classification.
        output_path: Filepath destination for the CSV export.

    Returns:
        pd.DataFrame containing the exported scored transactions.
    """
    test_slice = engineered_df.loc[test_indices].copy()

    # Generate standardized transaction IDs
    test_slice["transaction_id"] = [f"TXN_{i:06d}" for i in range(len(test_slice))]
    test_slice["fraud_probability"] = fraud_probabilities
    test_slice["predicted_label"] = (fraud_probabilities >= operating_threshold).astype(int)
    test_slice["actual_label"] = y_test.values

    # Handle Amount column naming
    if "Amount" in test_slice.columns and "amount" not in test_slice.columns:
        test_slice["amount"] = test_slice["Amount"]
    elif "amount" not in test_slice.columns:
        test_slice["amount"] = 0.0

    # Ensure merchant_category exists
    if "merchant_category" not in test_slice.columns:
        test_slice["merchant_category"] = "General"

    # Ensure hour_of_day exists
    if "hour_of_day" not in test_slice.columns:
        test_slice["hour_of_day"] = 0.0

    export_df = test_slice[[
        "transaction_id",
        "amount",
        "merchant_category",
        "hour_of_day",
        "fraud_probability",
        "predicted_label",
        "actual_label",
    ]]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    export_df.to_csv(out_file, index=False)
    print(f"Scored transactions saved ({len(export_df):,} rows) -> '{out_file.resolve()}'")

    return export_df


def export_threshold_metrics(
    cost_df: pd.DataFrame,
    output_path: Union[str, Path] = "outputs/threshold_metrics.csv",
) -> None:
    """
    Export threshold analysis metrics for Power BI threshold sweep visualization.

    Args:
        cost_df: DataFrame output from compute_threshold_metrics.
        output_path: Destination CSV path.
    """
    export_cols = [
        "threshold",
        "precision",
        "recall",
        "f1_score",
        "total_cost",
        "cost_per_1000",
        "tp_count",
        "fp_count",
        "fn_count",
        "tn_count",
    ]
    available_cols = [c for c in export_cols if c in cost_df.columns]

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    cost_df[available_cols].to_csv(out_file, index=False)
    print(f"Threshold metrics saved ({len(cost_df)} rows) -> '{out_file.resolve()}'")


def export_pr_curve_points(
    pr_curve_df: pd.DataFrame,
    output_path: Union[str, Path] = "outputs/pr_curve_points.csv",
) -> None:
    """
    Export precision-recall curve coordinates.

    Args:
        pr_curve_df: DataFrame with 'threshold', 'precision', 'recall'.
        output_path: Destination CSV path.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pr_curve_df.to_csv(out_file, index=False)
    print(f"PR curve points saved ({len(pr_curve_df)} points) -> '{out_file.resolve()}'")


def export_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Union[str, Path] = "outputs/feature_importance.csv",
) -> None:
    """
    Export SHAP feature importance rankings.

    Args:
        importance_df: DataFrame with 'feature_name' and 'importance_score'.
        output_path: Destination CSV path.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(out_file, index=False)
    print(f"Feature importance saved ({len(importance_df)} features) -> '{out_file.resolve()}'")


def export_all_dashboard_artifacts(
    engineered_df: pd.DataFrame,
    test_indices: pd.Index,
    y_test: pd.Series,
    fraud_probabilities: np.ndarray,
    operating_threshold: float,
    cost_df: pd.DataFrame,
    pr_curve_df: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir: Union[str, Path] = "outputs",
) -> Dict[str, str]:
    """
    Export all four dashboard artifacts simultaneously.

    Returns:
        Dictionary mapping artifact names to their filepaths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scored_file = out_path / "scored_transactions.csv"
    threshold_file = out_path / "threshold_metrics.csv"
    pr_file = out_path / "pr_curve_points.csv"
    importance_file = out_path / "feature_importance.csv"

    export_scored_transactions(
        engineered_df=engineered_df,
        test_indices=test_indices,
        y_test=y_test,
        fraud_probabilities=fraud_probabilities,
        operating_threshold=operating_threshold,
        output_path=scored_file,
    )
    export_threshold_metrics(cost_df, threshold_file)
    export_pr_curve_points(pr_curve_df, pr_file)
    export_feature_importance(importance_df, importance_file)

    return {
        "scored_transactions": str(scored_file.resolve()),
        "threshold_metrics": str(threshold_file.resolve()),
        "pr_curve_points": str(pr_file.resolve()),
        "feature_importance": str(importance_file.resolve()),
    }


def print_final_scorecard(
    model_name: str,
    scale_pos_weight: float,
    pr_auc: float,
    roc_auc: float,
    optimal_threshold: float,
    precision: float,
    recall: float,
    cost_per_1000: float,
    total_flagged: int,
) -> None:
    """
    Print formatted terminal summary scorecard of final model performance.
    """
    print(f"\n{'='*55}")
    print("FINAL MODEL SCORECARD")
    print(f"{'='*55}")
    print(f"Model:             {model_name}")
    print(f"Imbalance:         scale_pos_weight = {scale_pos_weight:.1f}")
    print(f"PR-AUC:            {pr_auc:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.2f} (cost-sensitive)")
    print(f"Precision:         {precision:.4f}")
    print(f"Recall:            {recall:.4f}")
    print(f"Cost/1000 tx:      ₹{cost_per_1000:.2f}")
    print(f"Total Flagged:     {total_flagged:,}")
    print(f"{'='*55}\n")
