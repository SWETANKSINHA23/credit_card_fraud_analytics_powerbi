from pathlib import Path
import joblib
import numpy as np
import pandas as pd


FEATURE_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_of_day"]


class FraudModelService:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent.parent
        models_dir = base_dir / "models"

        self.lgbm_model = joblib.load(models_dir / "lgbm_model.pkl")
        self.rf_model = joblib.load(models_dir / "rf_model.pkl")
        self.meta_learner = joblib.load(models_dir / "meta_learner.pkl")

        self.threshold = 0.98
        self.model_type = "Stacked Ensemble (LightGBM + RandomForest -> LogisticRegression)"
        self.model_version = "1.0"
        self.training_roc_auc = 0.9817
        self.training_pr_auc = 0.8491

    def _to_dataframe(self, records):
        df = pd.DataFrame(records)
        df = df[FEATURE_COLUMNS]
        return df

    def predict_proba(self, records):
        df = self._to_dataframe(records)

        lgbm_probs = self.lgbm_model.predict_proba(df)[:, 1]
        rf_probs = self.rf_model.predict_proba(df)[:, 1]

        meta_features = np.column_stack([lgbm_probs, rf_probs])
        final_probs = self.meta_learner.predict_proba(meta_features)[:, 1]

        return final_probs

    def predict(self, records):
        probs = self.predict_proba(records)
        labels = (probs >= self.threshold).astype(int)
        return probs, labels

    def get_model_info(self):
        return {
            "model_type": self.model_type,
            "model_version": self.model_version,
            "threshold": self.threshold,
            "training_roc_auc": self.training_roc_auc,
            "training_pr_auc": self.training_pr_auc,
            "features": FEATURE_COLUMNS,
        }


model_service = FraudModelService()