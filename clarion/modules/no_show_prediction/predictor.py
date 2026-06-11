"""Serving-side predictor for Module M3 no-show prediction.

Loads the joblib bundle the trainer wrote, then scores one
appointment dict at a time. The output is the same NoShowPrediction
wire shape downstream consumers see in ``predictions.jsonl`` — no
extra translation layer.

The predictor is deliberately stateless once loaded; ``predict_one``
does the row-encoding -> proba -> risk_band roundtrip in one call so
the front-desk UI can show a colour band on hover without further
business logic.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from clarion.modules.no_show_prediction.dataset import FEATURE_COLUMNS, encode_row
from clarion.schemas import NoShowModelMetadata, NoShowPrediction, NoShowRiskBand

# Risk-band thresholds — locked in code so the on-disk band and the
# UI's colour map can never drift apart. Boundaries are exclusive on
# the low side, inclusive on the high side:
#   p_no_show < 0.25            -> low
#   0.25 <= p_no_show < 0.50    -> medium
#   p_no_show >= 0.50           -> high
RISK_BAND_MEDIUM_THRESHOLD = 0.25
RISK_BAND_HIGH_THRESHOLD = 0.50


def risk_band_for(p_no_show: float) -> NoShowRiskBand:
    """Map a probability to its colour-coded risk band."""
    if p_no_show >= RISK_BAND_HIGH_THRESHOLD:
        return "high"
    if p_no_show >= RISK_BAND_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


class NoShowPredictor:
    """Lazy-loaded XGBoost wrapper.

    Instantiate once per process with ``NoShowPredictor.load(path)``,
    then call ``predict_one`` per appointment. The booster + metadata
    are immutable after load.
    """

    def __init__(self, model: Any, metadata: NoShowModelMetadata) -> None:
        # Sanity check — if the persisted feature layout doesn't match
        # what the dataset module currently produces, refuse to score.
        # A silent drift here is the worst kind of bug (predictions
        # look fine but are aligned to the wrong columns).
        if tuple(metadata.feature_columns) != FEATURE_COLUMNS:
            raise ValueError(
                "feature column drift between persisted model and dataset module — "
                "retrain before serving"
            )
        self._model = model
        self._metadata = metadata

    @classmethod
    def load(cls, path: Path) -> NoShowPredictor:
        """Reload a trainer-written bundle from disk."""
        bundle = joblib.load(path)
        metadata = NoShowModelMetadata.model_validate(bundle["metadata"])
        return cls(model=bundle["model"], metadata=metadata)

    @property
    def metadata(self) -> NoShowModelMetadata:
        return self._metadata

    @property
    def model_version(self) -> str:
        return self._metadata.model_version

    def predict_proba_batch(self, X: NDArray[np.float32]) -> NDArray[np.float64]:
        """Batch-score a pre-encoded feature matrix.

        Used by the evaluation metric to score a held-out test set
        without paying the encode_row roundtrip per row.
        """
        return self._model.predict_proba(X)[:, 1].astype(np.float64)  # type: ignore[no-any-return]

    def predict_one(
        self,
        features: dict[str, float | str | int],
        *,
        customer_id: str,
        appointment_id: str,
        patient_id: str | None = None,
    ) -> NoShowPrediction:
        """Score one appointment and wrap the result in the wire shape.

        ``features`` carries the seven raw fields the dataset module
        knows about (lead_time_days, prior_no_show_rate,
        is_new_patient, day_of_week, payer, age_band,
        appointment_type). ``encode_row`` raises on unknown
        categorical values.
        """
        vec = encode_row(features).reshape(1, -1)
        proba = self._model.predict_proba(vec)[0, 1]
        p = float(np.clip(proba, 0.0, 1.0))
        return NoShowPrediction(
            customer_id=customer_id,
            appointment_id=appointment_id,
            patient_id=patient_id,
            generated_at=datetime.now(UTC),
            model_version=self._metadata.model_version,
            p_no_show=round(p, 4),
            risk_band=risk_band_for(p),
        )
