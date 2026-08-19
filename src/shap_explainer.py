"""
SHAP Explainability and Feature Importance Module.

Provides model interpretability and auditability for the fraud detection system
using Shapley Additive Explanations (SHAP TreeExplainer on LightGBM).

Generates:
1. Feature importance rankings based on mean absolute SHAP values.
2. Visual summary bar charts exported to docs/.
3. Structured CSV artifacts for Power BI dashboard consumption.
"""

from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_importance(
    model: Any,
    X_sample: pd.DataFrame,
    feature_names: Optional[List[str]] = None,
    max_display: int = 15,
    output_plot_path: Optional[Union[str, Path]] = "docs/shap_importance.png",
    output_csv_path: Optional[Union[str, Path]] = "outputs/feature_importance.csv",
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Compute SHAP values for the LightGBM base model and generate importance rankings.

    Args:
        model: Fitted tree-based model (e.g. LGBMClassifier).
        X_sample: Sampled feature matrix for explanation (typically 1,000-2,000 instances).
        feature_names: Column names for features.
        max_display: Number of top features to include in visual summary plot.
        output_plot_path: Destination path for SHAP bar plot image.
        output_csv_path: Destination path for feature importance CSV.

    Returns:
        Tuple of (importance_dataframe, raw_positive_class_shap_values).
    """
    if feature_names is None:
        feature_names = list(X_sample.columns)

    print("Initializing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # For binary classification, extract positive class (Class=1) SHAP values
    if isinstance(shap_values, list):
        shap_values_pos = shap_values[1]
    else:
        shap_values_pos = shap_values

    # Calculate mean absolute impact on model output magnitude
    mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature_name": feature_names,
        "importance_score": mean_abs_shap,
    }).sort_values("importance_score", ascending=False).reset_index(drop=True)

    # Generate and save SHAP summary bar chart
    if output_plot_path is not None:
        fig = plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_pos,
            X_sample,
            feature_names=feature_names,
            plot_type="bar",
            show=False,
            max_display=max_display,
            plot_size=(10, 6),
        )
        plt.title("SHAP Feature Importance — LightGBM Base Model", fontsize=12, fontweight="bold")
        plt.tight_layout()

        plot_path = Path(output_plot_path)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"SHAP summary plot saved to '{plot_path.resolve()}'")

    # Save feature importance CSV for Power BI
    if output_csv_path is not None:
        csv_path = Path(output_csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(csv_path, index=False)
        print(f"Feature importance saved to '{csv_path.resolve()}'")

    return importance_df, shap_values_pos
