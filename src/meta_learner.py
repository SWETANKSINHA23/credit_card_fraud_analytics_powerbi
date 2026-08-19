"""
Stacked Ensemble and Meta-Learner Training Module.

Implements the Level-1 Meta-Learner (Logistic Regression) stacked on top of
Level-0 predictions (LightGBM + Random Forest).
Prevents target leakage by generating out-of-fold (OOF) cross-validated
probabilities during training.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score

from src.base_models import get_lightgbm_classifier, get_random_forest_classifier


def get_meta_learner(
    max_iter: int = 1000,
    class_weight: str = "balanced",
    random_state: int = 42,
    **kwargs: Any,
) -> LogisticRegression:
    """
    Construct the Level-1 Logistic Regression meta-learner.

    Args:
        max_iter: Maximum iterations for convergence.
        class_weight: Balanced class weighting to handle imbalance.
        random_state: Random seed for solver reproducibility.
        **kwargs: Additional parameters for LogisticRegression.

    Returns:
        Configured LogisticRegression instance.
    """
    params: Dict[str, Any] = {
        "max_iter": max_iter,
        "class_weight": class_weight,
        "random_state": random_state,
    }
    params.update(kwargs)
    return LogisticRegression(**params)


def build_stacked_ensemble(
    scale_pos_weight: Optional[float] = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> StackingClassifier:
    """
    Build the full scikit-learn StackingClassifier pipeline.

    Args:
        scale_pos_weight: Imbalance weight for the LightGBM base model.
        n_splits: Number of stratified cross-validation folds for OOF predictions.
        random_state: Random seed for fold partitioning.

    Returns:
        Configured StackingClassifier instance.
    """
    lgbm_base = get_lightgbm_classifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    rf_base = get_random_forest_classifier(random_state=random_state)
    meta_learner = get_meta_learner(random_state=random_state)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    stacked_classifier = StackingClassifier(
        estimators=[
            ("lgbm", lgbm_base),
            ("rf", rf_base),
        ],
        final_estimator=meta_learner,
        cv=cv,
        stack_method="predict_proba",
        n_jobs=-1,
        passthrough=False,
    )

    return stacked_classifier


def train_stacked_pipeline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: Optional[float] = None,
    n_splits: int = 5,
    random_state: int = 42,
    models_dir: Optional[Union[str, Path]] = "models",
) -> Tuple[StackingClassifier, Dict[str, Any]]:
    """
    Train the complete stacked ensemble and individual base models,
    then save serialised artifacts to disk.

    Args:
        X_train: Training feature set.
        y_train: Training label set.
        scale_pos_weight: Cost-sensitive weight for positive class.
        n_splits: Number of CV splits.
        random_state: Random seed.
        models_dir: Directory path to persist .pkl artifacts.

    Returns:
        Tuple of (trained_stacked_model, dict_of_all_models).
    """
    print("Building Stacked Ensemble Pipeline...")
    stacked_model = build_stacked_ensemble(
        scale_pos_weight=scale_pos_weight,
        n_splits=n_splits,
        random_state=random_state,
    )

    print("Fitting StackingClassifier on training data...")
    stacked_model.fit(X_train, y_train)

    # Train standalone base models for modular inference / SHAP
    print("Fitting standalone LightGBM and Random Forest base models...")
    lgbm_standalone = get_lightgbm_classifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
    )
    lgbm_standalone.fit(X_train, y_train)

    rf_standalone = get_random_forest_classifier(random_state=random_state)
    rf_standalone.fit(X_train, y_train)

    all_models = {
        "stacked_model": stacked_model,
        "lgbm_model": lgbm_standalone,
        "rf_model": rf_standalone,
        "meta_learner": stacked_model.final_estimator_,
    }

    if models_dir is not None:
        save_path = Path(models_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(stacked_model, save_path / "stacked_model.pkl")
        joblib.dump(lgbm_standalone, save_path / "lgbm_model.pkl")
        joblib.dump(rf_standalone, save_path / "rf_model.pkl")
        joblib.dump(stacked_model.final_estimator_, save_path / "meta_learner.pkl")
        print(f"Models successfully saved to '{save_path.resolve()}'")

    return stacked_model, all_models


def evaluate_stacked_ensemble(
    stacked_model: StackingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evaluate the final stacked ensemble on test data.

    Args:
        stacked_model: Fitted StackingClassifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary with PR-AUC and ROC-AUC metrics.
    """
    y_probs = stacked_model.predict_proba(X_test)[:, 1]
    pr_auc = average_precision_score(y_test, y_probs)
    roc_auc = roc_auc_score(y_test, y_probs)

    return {
        "PR-AUC": round(float(pr_auc), 4),
        "ROC-AUC": round(float(roc_auc), 4),
    }
