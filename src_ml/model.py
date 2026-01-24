"""
Classical ML models for protein-protein interaction prediction.

This module provides wrapper classes for:
- RandomForestClassifier
- XGBoostClassifier

Both use mean-pooled ESM embeddings concatenated for protein pairs.
Feature vector: [mean(emb_A), mean(emb_B)] → 3072-dim (for ESM-3 1536-dim embeddings).

These are NOT PyTorch modules. They use sklearn/xgboost APIs:
- fit(X, y)
- predict(X)
- predict_proba(X)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

try:
    from sklearn.ensemble import RandomForestClassifier as SklearnRF
except ImportError:
    SklearnRF = None  # type: ignore

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore


class RandomForest:
    """
    Random Forest classifier wrapper for PPI prediction.

    Uses sklearn's RandomForestClassifier under the hood.
    """

    name: str = "random_forest"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = 12,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = "sqrt",
        n_jobs: int = -1,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
        verbose: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Random Forest model.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of trees (None for unlimited).
            min_samples_split: Minimum samples required to split a node.
            min_samples_leaf: Minimum samples required at a leaf node.
            max_features: Number of features to consider for best split.
            n_jobs: Number of parallel jobs (-1 for all CPUs).
            random_state: Random seed for reproducibility.
            class_weight: Class weight strategy ("balanced" or None).
            verbose: Verbosity level.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        if SklearnRF is None:
            raise ImportError(
                "scikit-learn is required for RandomForest. "
                "Install with: pip install scikit-learn"
            )

        self.model = SklearnRF(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight=class_weight,
            verbose=verbose,
        )
        self._is_fitted = False

    def fit(self, X: NDArray[np.float32], y: NDArray[np.int64]) -> "RandomForest":
        """
        Fit the model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        logging.info(
            f"Training RandomForest on {X.shape[0]} samples, {X.shape[1]} features"
        )
        self.model.fit(X, y)
        self._is_fitted = True
        logging.info("RandomForest training complete")
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """
        Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, 2) for [neg, pos].
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return self.model.get_params()


class XGBoost:
    """
    XGBoost classifier wrapper for PPI prediction.

    Uses xgboost's XGBClassifier under the hood.
    """

    name: str = "xgboost"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        gamma: float = 0.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        scale_pos_weight: Optional[float] = None,
        n_jobs: int = -1,
        random_state: int = 42,
        use_label_encoder: bool = False,
        eval_metric: str = "logloss",
        verbosity: int = 1,
        early_stopping_rounds: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize XGBoost model.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum tree depth.
            learning_rate: Boosting learning rate (eta).
            subsample: Subsample ratio of training instances.
            colsample_bytree: Subsample ratio of columns for each tree.
            min_child_weight: Minimum sum of instance weight in a child.
            gamma: Minimum loss reduction for further partition.
            reg_alpha: L1 regularization term.
            reg_lambda: L2 regularization term.
            scale_pos_weight: Balance of positive/negative weights.
            n_jobs: Number of parallel threads (-1 for all).
            random_state: Random seed for reproducibility.
            use_label_encoder: Whether to use label encoder (deprecated).
            eval_metric: Evaluation metric.
            verbosity: Verbosity level (0: silent, 1: warning, 2: info).
            early_stopping_rounds: Stop if no improvement for N rounds.
            **kwargs: Additional arguments (ignored for compatibility).
        """
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is required for XGBoost model. "
                "Install with: pip install xgboost"
            )

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            use_label_encoder=use_label_encoder,
            eval_metric=eval_metric,
            verbosity=verbosity,
            early_stopping_rounds=early_stopping_rounds,
        )
        self._is_fitted = False

    def fit(
        self,
        X: NDArray[np.float32],
        y: NDArray[np.int64],
        eval_set: Optional[list] = None,
        early_stopping_rounds: Optional[int] = None,
    ) -> "XGBoost":
        """
        Fit the model to training data.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,).
            eval_set: List of (X, y) tuples for early stopping evaluation.
            early_stopping_rounds: Ignored (pass to __init__ instead).

        Returns:
            Self for method chaining.
        """
        logging.info(f"Training XGBoost on {X.shape[0]} samples, {X.shape[1]} features")

        fit_kwargs: Dict[str, Any] = {}
        if eval_set is not None:
            fit_kwargs["eval_set"] = eval_set
        # Note: early_stopping_rounds must be passed to constructor in newer XGBoost

        self.model.fit(X, y, **fit_kwargs)
        self._is_fitted = True
        logging.info("XGBoost training complete")
        return self

    def predict(self, X: NDArray[np.float32]) -> NDArray[np.int64]:
        """
        Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, 2) for [neg, pos].
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict_proba(X)

    def get_params(self) -> Dict[str, Any]:
        """Return model parameters."""
        return self.model.get_params()


# Model registry for factory pattern
ML_MODELS: Dict[str, type] = {
    "random_forest": RandomForest,
    "xgboost": XGBoost,
}


def build_ml_model(model_name: str, **kwargs: Any) -> RandomForest | XGBoost:
    """
    Factory function to build ML models.

    Args:
        model_name: One of "random_forest" or "xgboost".
        **kwargs: Model-specific parameters.

    Returns:
        Instantiated model.

    Raises:
        ValueError: If model_name is not recognized.
    """
    if model_name not in ML_MODELS:
        raise ValueError(
            f"Unknown ML model: '{model_name}'. Supported: {list(ML_MODELS.keys())}"
        )
    return ML_MODELS[model_name](**kwargs)
