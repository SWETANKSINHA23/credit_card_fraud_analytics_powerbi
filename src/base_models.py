"""
Base Level Classifiers Module for Fraud Detection Ensemble.

Defines and trains the heterogeneous Level-0 base models:
1. LightGBM: High-speed gradient-boosted decision trees with leaf-wise growth.
2. Random Forest: Bagged ensemble of decision trees with balanced class weighting.
"""

from typing import Any, Dict, Optional
import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def get_lightgbm_classifier(
    scale_pos_weight: Optional[float] = None,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    min_child_samples: int = 20,
    feature_fraction: float = 0.8,
    bagging_fraction: float = 0.8,
    bagging_freq: int = 5,
    n_estimators: int = 300,
    random_state: int = 42,
    **kwargs: Any,
) -> lgb.LGBMClassifier:
    """
    Construct a tuned LightGBM classifier for imbalanced fraud classification.

    Args:
        scale_pos_weight: Weight for positive (fraud) instances.
        learning_rate: Boosting learning rate.
        num_leaves: Maximum tree leaves for base learners.
        min_child_samples: Minimum number of data points needed in a child/leaf.
        feature_fraction: Subsample ratio of columns when constructing each tree.
        bagging_fraction: Subsample ratio of training data.
        bagging_freq: Frequency of bagging.
        n_estimators: Number of boosting iterations.
        random_state: Random number seed.
        **kwargs: Additional parameters passed to LGBMClassifier.

    Returns:
        Configured LGBMClassifier instance.
    """
    params: Dict[str, Any] = {
        "objective": "binary",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "min_child_samples": min_child_samples,
        "feature_fraction": feature_fraction,
        "bagging_fraction": bagging_fraction,
        "bagging_freq": bagging_freq,
        "n_estimators": n_estimators,
        "random_state": random_state,
        "verbose": -1,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    params.update(kwargs)
    return lgb.LGBMClassifier(**params)


def get_random_forest_classifier(
    n_estimators: int = 200,
    max_depth: int = 12,
    min_samples_leaf: int = 5,
    class_weight: str = "balanced",
    random_state: int = 42,
    n_jobs: int = -1,
    **kwargs: Any,
) -> RandomForestClassifier:
    """
    Construct a tuned Random Forest classifier for robust bagging predictions.

    Args:
        n_estimators: Number of trees in forest.
        max_depth: Maximum depth of the tree.
        min_samples_leaf: Minimum number of samples required at a leaf node.
        class_weight: Strategy for adjusting weights inversely proportional to class frequencies.
        random_state: Random seed for reproducibility.
        n_jobs: Number of parallel jobs.
        **kwargs: Additional parameters passed to RandomForestClassifier.

    Returns:
        Configured RandomForestClassifier instance.
    """
    params: Dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "class_weight": class_weight,
        "random_state": random_state,
        "n_jobs": n_jobs,
    }
    params.update(kwargs)
    return RandomForestClassifier(**params)


def train_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: Optional[float] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Instantiate and train both LightGBM and Random Forest base models.

    Args:
        X_train: Training features.
        y_train: Training labels.
        scale_pos_weight: Cost-sensitive weight for LightGBM.
        random_state: Random seed.

    Returns:
        Dictionary mapping model names to fitted model instances.
    """
    print("Training Level-0 Base Model: LightGBM...")
    lgbm_model = get_lightgbm_classifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    lgbm_model.fit(X_train, y_train)

    print("Training Level-0 Base Model: Random Forest...")
    rf_model = get_random_forest_classifier(random_state=random_state)
    rf_model.fit(X_train, y_train)

    return {
        "lgbm": lgbm_model,
        "rf": rf_model,
    }


def evaluate_base_models(
    models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Evaluate fitted base models on a test set.

    Args:
        models: Dictionary of model name -> fitted model.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        pd.DataFrame comparing PR-AUC, ROC-AUC, and F1 score.
    """
    records = []
    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        records.append({
            "Model": name,
            "PR-AUC": round(float(average_precision_score(y_test, probs)), 4),
            "ROC-AUC": round(float(roc_auc_score(y_test, probs)), 4),
            "F1@0.5": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        })

    return pd.DataFrame(records)
