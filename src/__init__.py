"""
Credit Card Fraud Analytics & Detection Package.

Modules:
- data_loader: Data ingestion, feature engineering, and stratified splitting.
- imbalance_strategies: Benchmarking SMOTE, ADASYN, and cost-sensitive class weighting.
- base_models: Level-0 LightGBM and Random Forest classifiers.
- meta_learner: Level-1 Logistic Regression meta-learner and StackingClassifier.
- threshold_optimizer: Cost-sensitive threshold search and Precision-Recall analysis.
- shap_explainer: SHAP TreeExplainer feature importance and visualizations.
- export: Power BI dashboard CSV artifact export and scorecard reporting.
"""

from src.base_models import (
    evaluate_base_models,
    get_lightgbm_classifier,
    get_random_forest_classifier,
    train_base_models,
)
from src.data_loader import (
    engineer_features,
    get_feature_columns,
    load_and_preprocess_pipeline,
    load_raw_data,
    split_data,
)
from src.export import (
    export_all_dashboard_artifacts,
    export_feature_importance,
    export_pr_curve_points,
    export_scored_transactions,
    export_threshold_metrics,
    print_final_scorecard,
)
from src.imbalance_strategies import (
    compare_imbalance_strategies,
    compute_scale_pos_weight,
    evaluate_resampling_strategy,
)
from src.meta_learner import (
    build_stacked_ensemble,
    evaluate_stacked_ensemble,
    get_meta_learner,
    train_stacked_pipeline,
)
from src.shap_explainer import compute_shap_importance
from src.threshold_optimizer import (
    compute_pr_curve_dataframe,
    compute_threshold_metrics,
    find_optimal_thresholds,
    plot_and_save_pr_cost_curves,
)

__all__ = [
    # Data Loader
    "load_raw_data",
    "engineer_features",
    "split_data",
    "load_and_preprocess_pipeline",
    "get_feature_columns",
    # Imbalance Strategies
    "compute_scale_pos_weight",
    "evaluate_resampling_strategy",
    "compare_imbalance_strategies",
    # Base Models
    "get_lightgbm_classifier",
    "get_random_forest_classifier",
    "train_base_models",
    "evaluate_base_models",
    # Meta Learner
    "get_meta_learner",
    "build_stacked_ensemble",
    "train_stacked_pipeline",
    "evaluate_stacked_ensemble",
    # Threshold Optimizer
    "compute_threshold_metrics",
    "find_optimal_thresholds",
    "compute_pr_curve_dataframe",
    "plot_and_save_pr_cost_curves",
    # SHAP Explainer
    "compute_shap_importance",
    # Export
    "export_scored_transactions",
    "export_threshold_metrics",
    "export_pr_curve_points",
    "export_feature_importance",
    "export_all_dashboard_artifacts",
    "print_final_scorecard",
]
