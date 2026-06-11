"""XGBoost trainer for Module M3 no-show prediction.

Pipeline:

    df = generate_dataset(seed)
    result = train(df, seed=seed)
    persist(result, models/no_show_v1.joblib)

The persisted bundle is a single joblib file containing both the
fitted booster and a NoShowModelMetadata block — the predictor
reloads both in one call. We deliberately do not version the
on-disk file name by hash; ``metadata.model_version`` is the
audit handle that ends up on every NoShowPrediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from clarion.modules.no_show_prediction.dataset import FEATURE_COLUMNS, Dataset
from clarion.modules.no_show_prediction.metric import compute_top_decile_lift
from clarion.schemas import NoShowModelMetadata

# Bumped whenever the training pipeline or feature layout changes.
MODEL_VERSION = "no_show_v1"

# Conservative XGBoost hyperparameters — small forest, shallow trees,
# regularization tuned by hand to avoid overfit on n=2000.
_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.08,
    "reg_lambda": 1.0,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs": 1,
}


@dataclass(frozen=True)
class TrainResult:
    """Bundle returned by ``train`` and consumed by ``persist``."""

    model: XGBClassifier
    metadata: NoShowModelMetadata


def train(
    dataset: Dataset,
    *,
    seed: int,
    n_splits: int = 5,
    params: dict[str, Any] | None = None,
) -> TrainResult:
    """Fit an XGBoost classifier with 5-fold stratified CV scoring.

    The returned booster is fit on the **full** dataset; the CV pass
    only produces the ``roc_auc_cv`` + ``top_decile_lift_cv`` numbers
    that get stamped into the metadata. That mirrors how a real PMS
    team would ship — pick hyperparameters with CV, then train on
    everything you've got before deploying.
    """
    cfg = dict(_DEFAULT_PARAMS)
    if params:
        cfg.update(params)
    cfg["random_state"] = seed

    X, y = dataset.X, dataset.y

    # CV pass for honest metric reporting.
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    cv_aucs: list[float] = []
    cv_lifts: list[float] = []
    for fold_seed, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
        fold_params = dict(cfg)
        fold_params["random_state"] = seed + fold_seed
        fold_model = XGBClassifier(**fold_params)
        fold_model.fit(X[train_idx], y[train_idx], verbose=False)
        proba = fold_model.predict_proba(X[test_idx])[:, 1]
        cv_aucs.append(float(roc_auc_score(y[test_idx], proba)))
        cv_lifts.append(compute_top_decile_lift(y[test_idx], proba.astype(np.float64)))

    # Final fit on everything.
    final = XGBClassifier(**cfg)
    final.fit(X, y, verbose=False)

    metadata = NoShowModelMetadata(
        model_version=MODEL_VERSION,
        trained_at=datetime.now(UTC),
        n_train=int(len(y)),
        n_features=len(FEATURE_COLUMNS),
        roc_auc_cv=round(float(np.mean(cv_aucs)), 4),
        top_decile_lift_cv=round(float(np.mean(cv_lifts)), 4),
        feature_columns=list(FEATURE_COLUMNS),
        seed=seed,
    )
    return TrainResult(model=final, metadata=metadata)


def persist(result: TrainResult, path: Path) -> None:
    """Write the booster + metadata to a single joblib bundle.

    Parent dirs are created on demand so callers don't have to
    pre-mkdir ``models/``. The metadata is stored as the validated
    Pydantic dict so a reload can round-trip back through pydantic.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": result.model,
        "metadata": result.metadata.model_dump(mode="json"),
    }
    joblib.dump(bundle, path)


