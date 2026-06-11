"""Module M3 tests — dataset, trainer, predictor, metric."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from clarion.modules.no_show_prediction import (
    FEATURE_COLUMNS,
    MODEL_VERSION,
    RISK_BAND_HIGH_THRESHOLD,
    RISK_BAND_MEDIUM_THRESHOLD,
    NoShowPredictor,
    compute_no_show_metrics,
    compute_top_decile_lift,
    encode_row,
    generate_dataset,
    persist,
    risk_band_for,
    train,
)
from clarion.schemas import NoShowPrediction

# ---------- dataset ----------


def test_dataset_shape_matches_feature_columns() -> None:
    d = generate_dataset(seed=7, n=200)
    assert d.X.shape == (200, len(FEATURE_COLUMNS))
    assert d.y.shape == (200,)
    assert d.X.dtype == np.float32
    assert d.y.dtype == np.int8
    assert d.feature_columns == FEATURE_COLUMNS
    assert len(d.raw_rows) == 200


def test_dataset_is_deterministic_for_same_seed() -> None:
    a = generate_dataset(seed=7, n=200)
    b = generate_dataset(seed=7, n=200)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.y, b.y)
    assert a.raw_rows == b.raw_rows


def test_dataset_differs_across_seeds() -> None:
    a = generate_dataset(seed=7, n=200)
    b = generate_dataset(seed=8, n=200)
    # Different seed -> different draws (overwhelmingly likely).
    assert not np.array_equal(a.X, b.X)


def test_encode_row_unknown_categorical_raises() -> None:
    row = {
        "lead_time_days": 10,
        "prior_no_show_rate": 0.1,
        "is_new_patient": 0,
        "day_of_week": "blursday",  # not a real day
        "payer": "aetna",
        "age_band": "18-30",
        "appointment_type": "routine",
    }
    with pytest.raises(ValueError, match="day_of_week"):
        encode_row(row)


# ---------- trainer ----------


def test_train_returns_metadata_with_cv_metrics() -> None:
    d = generate_dataset(seed=7, n=600)
    r = train(d, seed=7, n_splits=3)
    assert r.metadata.model_version == MODEL_VERSION
    assert r.metadata.n_train == 600
    assert r.metadata.n_features == len(FEATURE_COLUMNS)
    assert 0.0 <= r.metadata.roc_auc_cv <= 1.0
    assert r.metadata.top_decile_lift_cv >= 0.0
    assert tuple(r.metadata.feature_columns) == FEATURE_COLUMNS
    assert r.metadata.seed == 7


def test_train_is_deterministic_given_seed() -> None:
    d1 = generate_dataset(seed=7, n=600)
    d2 = generate_dataset(seed=7, n=600)
    r1 = train(d1, seed=7, n_splits=3)
    r2 = train(d2, seed=7, n_splits=3)
    # Same data + same seed -> same booster + same CV metric.
    assert r1.metadata.roc_auc_cv == r2.metadata.roc_auc_cv
    assert r1.metadata.top_decile_lift_cv == r2.metadata.top_decile_lift_cv


# ---------- predictor ----------


def test_persist_load_predict_roundtrip(tmp_path: Path) -> None:
    d = generate_dataset(seed=7, n=600)
    r = train(d, seed=7, n_splits=3)
    bundle = tmp_path / "model.joblib"
    persist(r, bundle)
    assert bundle.is_file()

    predictor = NoShowPredictor.load(bundle)
    assert predictor.model_version == MODEL_VERSION
    assert predictor.metadata.n_train == 600

    out = predictor.predict_one(
        d.raw_rows[0],
        customer_id="ophthalmology",
        appointment_id="appt_001",
        patient_id="pat_001",
    )
    assert isinstance(out, NoShowPrediction)
    assert 0.0 <= out.p_no_show <= 1.0
    assert out.risk_band in {"low", "medium", "high"}
    assert out.model_version == MODEL_VERSION
    assert out.schema_version == "1.0.0"


def test_risk_band_thresholds() -> None:
    assert risk_band_for(0.0) == "low"
    assert risk_band_for(RISK_BAND_MEDIUM_THRESHOLD - 1e-6) == "low"
    assert risk_band_for(RISK_BAND_MEDIUM_THRESHOLD) == "medium"
    assert risk_band_for(RISK_BAND_HIGH_THRESHOLD - 1e-6) == "medium"
    assert risk_band_for(RISK_BAND_HIGH_THRESHOLD) == "high"
    assert risk_band_for(1.0) == "high"


def test_predictor_rejects_drifted_feature_columns(tmp_path: Path) -> None:
    """If the persisted metadata's feature_columns disagrees with the
    dataset module's current FEATURE_COLUMNS, NoShowPredictor refuses
    to load. That's the contract: silent column drift -> wrong
    predictions on right-looking rows."""
    d = generate_dataset(seed=7, n=400)
    r = train(d, seed=7, n_splits=3)
    bundle = tmp_path / "model.joblib"
    persist(r, bundle)

    import joblib

    loaded = joblib.load(bundle)
    loaded["metadata"]["feature_columns"] = ["mystery_feature", *FEATURE_COLUMNS[1:]]
    joblib.dump(loaded, bundle)

    with pytest.raises(ValueError, match="feature column drift"):
        NoShowPredictor.load(bundle)


# ---------- metric ----------


def test_top_decile_lift_baseline_is_one_for_random_scores() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=2000).astype(np.int8)
    s = rng.random(2000)  # uncorrelated with y
    lift = compute_top_decile_lift(y, s)
    # Random scores should give lift ~= 1.0 — be generous on tolerance.
    assert 0.7 <= lift <= 1.3


def test_top_decile_lift_perfect_ranking_is_max() -> None:
    # If scores perfectly rank no-shows first, top decile = 100% positives;
    # lift = 1 / base_rate. Construct base_rate = 0.2 so expected lift = 5.0.
    y = np.array([1] * 200 + [0] * 800, dtype=np.int8)
    # Higher score for positives.
    s = np.concatenate([np.full(200, 0.99), np.full(800, 0.01)])
    lift = compute_top_decile_lift(y, s)
    assert lift == pytest.approx(5.0, abs=1e-6)


def test_compute_no_show_metrics_returns_none_when_missing(tmp_path: Path) -> None:
    assert compute_no_show_metrics(tmp_path / "nope.joblib", seed=99) is None


def test_compute_no_show_metrics_uses_held_out_seed(tmp_path: Path) -> None:
    d = generate_dataset(seed=7, n=800)
    r = train(d, seed=7, n_splits=3)
    bundle = tmp_path / "model.joblib"
    persist(r, bundle)

    result = compute_no_show_metrics(bundle, seed=999, n_test=300)
    assert result is not None
    assert 0.0 <= result.roc_auc <= 1.0
    assert result.top_decile_lift >= 0.0
    assert result.n_test == 300
    assert result.model_version == MODEL_VERSION
