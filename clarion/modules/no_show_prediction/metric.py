"""Evaluation metrics for Module M3 no-show prediction.

Two callable surfaces:

* ``compute_top_decile_lift(y_true, y_score)`` — the same lift the
  trainer reports under CV, factored out so the reporter and the
  trainer share one definition. Lift > 1.0 means the top-10%
  scored cohort outperforms the base rate.
* ``compute_no_show_metrics(model_path, *, seed, n_test)`` — load
  the persisted predictor and score a freshly generated held-out
  set, returning roc_auc + top_decile_lift. None when the model
  file is missing (module disabled or never trained).

The held-out generation uses a seed different from the training
seed so it's a real out-of-fold measurement, not a re-roll of
training data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

from clarion.modules.no_show_prediction.dataset import generate_dataset
from clarion.modules.no_show_prediction.predictor import NoShowPredictor


@dataclass(frozen=True)
class NoShowEvalResult:
    """Held-out evaluation numbers folded into the report."""

    roc_auc: float
    top_decile_lift: float
    n_test: int
    model_version: str


def compute_top_decile_lift(
    y_true: NDArray[np.int8], y_score: NDArray[np.float64]
) -> float:
    """Positive rate among the top-10% scored cohort / base rate.

    Lift of 1.0 means the model has no signal; lift > 1.0 means the
    front desk would catch more no-shows by working the top decile
    than by calling everyone uniformly. Returns 0.0 when the base
    rate is zero (degenerate dataset).
    """
    n = len(y_true)
    if n == 0:
        return 0.0
    base_rate = float(np.mean(y_true))
    if base_rate <= 0.0:
        return 0.0
    k = max(1, n // 10)
    top_idx = np.argsort(y_score)[-k:]
    top_rate = float(np.mean(y_true[top_idx]))
    return round(top_rate / base_rate, 4)


def compute_no_show_metrics(
    model_path: Path,
    *,
    seed: int,
    n_test: int = 500,
) -> NoShowEvalResult | None:
    """Score a held-out synthetic test set through the persisted model.

    Returns None when ``model_path`` doesn't exist — that's how the
    reporter detects "module disabled" without taking a hard
    dependency on the customer-config layer.

    ``seed`` should differ from the training seed. The reporter
    enforces that by adding a fixed offset.
    """
    if not model_path.is_file():
        return None

    predictor = NoShowPredictor.load(model_path)
    test = generate_dataset(seed=seed, n=n_test)
    scores = predictor.predict_proba_batch(test.X)

    roc_auc = round(float(roc_auc_score(test.y, scores)), 4)
    lift = compute_top_decile_lift(test.y, scores)

    return NoShowEvalResult(
        roc_auc=roc_auc,
        top_decile_lift=lift,
        n_test=int(len(test.y)),
        model_version=predictor.model_version,
    )
