"""Interpretability engine using TreeSHAP for operational factor attribution."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


class OperationalExplainer:
    """Provides global and patient-level operational factor attributions using SHAP."""

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

        # If model is CalibratedClassifierCV or FrozenEstimator, extract underlying base estimator
        estimator = model
        if hasattr(estimator, "calibrated_classifiers_") and len(estimator.calibrated_classifiers_) > 0:
            estimator = estimator.calibrated_classifiers_[0].estimator
        if type(estimator).__name__ == "FrozenEstimator" and hasattr(estimator, "estimator"):
            estimator = estimator.estimator
        self.base_estimator = estimator

        try:
            self.explainer = shap.TreeExplainer(self.base_estimator)
            logger.info("Initialized TreeExplainer successfully.")
        except Exception as e:
            logger.warning("TreeExplainer fallback to LinearExplainer or Explainer: %s", e)
            self.explainer = shap.Explainer(self.base_estimator)

    def get_global_importance(self, X_sample: np.ndarray, top_n: int = 15) -> pd.DataFrame:
        """Compute mean absolute SHAP values across sample to rank operational risk drivers."""
        shap_values = self.explainer.shap_values(X_sample)

        # For binary classification, select positive class (index 1) if returned as list/3D
        if isinstance(shap_values, list):
            vals = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = shap_values

        mean_abs_shap = np.mean(np.abs(vals), axis=0)

        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean Absolute SHAP (Impact)": np.round(mean_abs_shap, 4),
        }).sort_values(by="Mean Absolute SHAP (Impact)", ascending=False).reset_index(drop=True)

        return importance_df.head(top_n)

    def explain_single_encounter(
        self,
        x_single: np.ndarray,
        raw_row: Optional[pd.Series] = None,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        """Explain individual patient encounter score by identifying top positive and negative factors."""
        if len(x_single.shape) == 1:
            x_single = x_single.reshape(1, -1)

        shap_values = self.explainer.shap_values(x_single)

        if isinstance(shap_values, list):
            vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif len(shap_values.shape) == 3:
            vals = shap_values[0, :, 1]
        else:
            vals = shap_values[0]

        df_factors = pd.DataFrame({
            "Feature": self.feature_names,
            "SHAP_Value": vals,
        })

        # Top positive contributors (pushed risk higher)
        top_risk_drivers = (
            df_factors[df_factors["SHAP_Value"] > 0]
            .sort_values(by="SHAP_Value", ascending=False)
            .head(top_n)
            .to_dict(orient="records")
        )

        # Top protective contributors (lowered risk)
        top_protective_factors = (
            df_factors[df_factors["SHAP_Value"] < 0]
            .sort_values(by="SHAP_Value", ascending=True)
            .head(top_n)
            .to_dict(orient="records")
        )

        return {
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
            "raw_feature_count": len(self.feature_names),
        }
