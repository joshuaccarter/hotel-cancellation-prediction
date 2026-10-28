from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from hotel_cancellation_prediction.config import settings
from hotel_cancellation_prediction.data.loader import load_clean_data
from hotel_cancellation_prediction.features.pipeline import (
    build_preprocessor,
    create_features,
)
from hotel_cancellation_prediction.utils.logger import setup_logger

logger = setup_logger(__name__)


def get_default_artifact_path() -> Path:
    """Returns the absolute or configured path to the model artifact."""
    artifact_dir = settings.paths.model_artifacts_dir
    filename = settings.models.individual_cancellation.artifact_filename
    path = Path(artifact_dir) / filename
    if not path.is_absolute():
        # Resolve relative to project root
        project_root = Path(__file__).resolve().parents[3]
        path = project_root / path
    return path


class CancellationPredictor:
    """Production predictor for individual reservation cancellation probability using LightGBM."""

    def __init__(
        self,
        model: Optional[Any] = None,
        preprocessor: Optional[Any] = None,
        threshold: Optional[float] = None,
        auto_load: bool = True,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.threshold = threshold if threshold is not None else settings.models.individual_cancellation.threshold

        if auto_load and self.model is None:
            artifact_path = get_default_artifact_path()
            if artifact_path.is_file():
                self.load(artifact_path)

        if self.preprocessor is None and self.model is None:
            self.preprocessor = build_preprocessor()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CancellationPredictor":
        """Fits preprocessor and trains LightGBM model on booking data."""
        cfg = settings.models.individual_cancellation
        logger.info("Extracting features for training on %d samples...", len(X))
        X_feats = create_features(X)

        if self.preprocessor is None:
            self.preprocessor = build_preprocessor()

        logger.info("Fitting preprocessor pipeline...")
        X_transformed = self.preprocessor.fit_transform(X_feats)

        logger.info(
            "Fitting LightGBM model (n_estimators=%d, lr=%.2f, leaves=%d)...",
            cfg.n_estimators,
            cfg.learning_rate,
            cfg.num_leaves,
        )
        self.model = lgb.LGBMClassifier(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            random_state=cfg.random_state,
            verbose=-1,
            n_jobs=-1,
        )
        self.model.fit(X_transformed, y)
        logger.info("Model fitting complete.")
        return self

    def predict_proba(self, booking_data: pd.DataFrame | Dict[str, Any]) -> float:
        """Predicts cancellation probability for a single booking."""
        if isinstance(booking_data, dict):
            df = pd.DataFrame([booking_data])
        else:
            df = booking_data.copy()

        features = create_features(df)

        if self.model is None or self.preprocessor is None:
            # Fallback heuristic if trained model not loaded
            lead_time = float(features["lead_time"].iloc[0] if "lead_time" in features.columns else 30)
            baseline_prob = min(max(lead_time / 300.0, 0.1), 0.9)
            return round(baseline_prob, 4)

        transformed = self.preprocessor.transform(features)
        proba = self.model.predict_proba(transformed)[0, 1]
        return round(float(proba), 4)

    def predict_proba_batch(self, bookings_df: pd.DataFrame) -> pd.Series:
        """Predicts cancellation probability for a batch of bookings."""
        features = create_features(bookings_df)
        if self.model is None or self.preprocessor is None:
            lead_time = features.get("lead_time", pd.Series(30, index=bookings_df.index))
            return (lead_time / 300.0).clip(0.1, 0.9).round(4)

        transformed = self.preprocessor.transform(features)
        probs = self.model.predict_proba(transformed)[:, 1]
        return pd.Series(probs, index=bookings_df.index).round(4)

    def predict(self, booking_data: pd.DataFrame | Dict[str, Any]) -> Dict[str, Any]:
        """Returns cancellation probability, risk categorization, and cost-optimal binary classification."""
        prob = self.predict_proba(booking_data)
        is_canceled = int(prob >= self.threshold)
        if prob >= 0.60:
            risk_level = "HIGH"
        elif prob >= 0.35:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        return {
            "cancellation_probability": prob,
            "prediction": is_canceled,
            "risk_level": risk_level,
            "threshold_used": self.threshold,
        }

    def predict_batch(self, bookings_df: pd.DataFrame) -> list[Dict[str, Any]]:
        """Returns predictions for a batch of bookings."""
        probs = self.predict_proba_batch(bookings_df)
        results = []
        for p in probs:
            p_val = float(p)
            is_canceled = int(p_val >= self.threshold)
            risk_level = "HIGH" if p_val >= 0.60 else ("MODERATE" if p_val >= 0.35 else "LOW")
            results.append(
                {
                    "cancellation_probability": p_val,
                    "prediction": is_canceled,
                    "risk_level": risk_level,
                    "threshold_used": self.threshold,
                }
            )
        return results

    def save(self, filepath: Optional[str | Path] = None) -> Path:
        """Serializes model, preprocessor, and configuration to disk."""
        target_path = Path(filepath) if filepath else get_default_artifact_path()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model,
            "preprocessor": self.preprocessor,
            "threshold": self.threshold,
            "version": settings.app.version,
        }
        joblib.dump(payload, target_path)
        logger.info("Successfully serialized model artifact to %s", target_path)
        return target_path

    def load(self, filepath: Optional[str | Path] = None) -> "CancellationPredictor":
        """Loads model and preprocessor from disk."""
        source_path = Path(filepath) if filepath else get_default_artifact_path()
        if not source_path.is_file():
            raise FileNotFoundError(f"Model artifact not found at: {source_path}")

        payload = joblib.load(source_path)
        self.model = payload["model"]
        self.preprocessor = payload["preprocessor"]
        self.threshold = payload.get("threshold", settings.models.individual_cancellation.threshold)
        logger.info("Successfully loaded model artifact from %s (threshold=%.2f)", source_path, self.threshold)
        return self


def train_and_save_model(
    save_path: Optional[str | Path] = None,
) -> Tuple[CancellationPredictor, Dict[str, float]]:
    """End-to-end training pipeline for the individual LightGBM cancellation model."""
    logger.info("Starting production model training pipeline...")
    df = load_clean_data()
    X = create_features(df)
    y = df[settings.data.target_column]

    cfg = settings.models.individual_cancellation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.random_state, stratify=y
    )

    predictor = CancellationPredictor(auto_load=False, threshold=cfg.threshold)
    predictor.fit(X_train, y_train)

    # Evaluate out-of-sample metrics
    X_test_trans = predictor.preprocessor.transform(create_features(X_test))
    y_probs = predictor.model.predict_proba(X_test_trans)[:, 1]

    metrics = {
        "roc_auc": round(float(roc_auc_score(y_test, y_probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, y_probs)), 4),
        "brier_score": round(float(brier_score_loss(y_test, y_probs)), 4),
    }
    logger.info("Model evaluation results: %s", metrics)

    target_path = predictor.save(save_path)
    logger.info("Model saved successfully to %s", target_path)
    return predictor, metrics
