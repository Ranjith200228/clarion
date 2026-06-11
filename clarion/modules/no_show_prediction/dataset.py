"""Synthetic appointment dataset for Module M3 no-show prediction.

We never train on real PHI — the trainer always consumes synthetic
rows generated here. Features were picked to mirror what a real PMS
exposes:

* ``lead_time_days``       — booking lead time (1-90)
* ``day_of_week``          — categorical, mon..sun
* ``prior_no_show_rate``   — historical rate for that patient (0-1)
* ``payer``                — aetna / cigna / bcbs / medicare / self_pay
* ``age_band``             — 18-30 / 31-50 / 51-70 / 71+
* ``appointment_type``     — routine / urgent / follow_up / pre_op / new_patient
* ``is_new_patient``       — 0/1

The no-show label is a deterministic function of those features plus
gaussian noise — strong signal on ``prior_no_show_rate`` and
``lead_time_days``, smaller effects everywhere else. That gives the
XGBoost trainer something meaningful to fit (~0.85 ROC-AUC at the
default sample size) without leaking the target.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

# Categorical levels — these freeze the one-hot column order, which is
# the contract the trained model and the predictor both rely on.
DAYS_OF_WEEK: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
PAYERS: tuple[str, ...] = ("aetna", "cigna", "bcbs", "medicare", "self_pay")
AGE_BANDS: tuple[str, ...] = ("18-30", "31-50", "51-70", "71+")
APPOINTMENT_TYPES: tuple[str, ...] = (
    "routine",
    "urgent",
    "follow_up",
    "pre_op",
    "new_patient",
)

# Numeric features come first, then the one-hot blocks in the order
# above. The list is the contract the predictor uses to align an
# incoming feature dict with the booster's column layout.
NUMERIC_FEATURES: tuple[str, ...] = (
    "lead_time_days",
    "prior_no_show_rate",
    "is_new_patient",
)


def _one_hot_columns() -> list[str]:
    cols: list[str] = list(NUMERIC_FEATURES)
    for prefix, levels in (
        ("day_of_week", DAYS_OF_WEEK),
        ("payer", PAYERS),
        ("age_band", AGE_BANDS),
        ("appointment_type", APPOINTMENT_TYPES),
    ):
        cols.extend(f"{prefix}={lvl}" for lvl in levels)
    return cols


FEATURE_COLUMNS: tuple[str, ...] = tuple(_one_hot_columns())


@dataclass(frozen=True)
class Dataset:
    """A generated training set.

    ``X`` is post-one-hot — shape ``(n, len(FEATURE_COLUMNS))``. ``y``
    is the binary no-show label. ``raw_rows`` is the pre-encoded view
    used by tests and the predictor's smoke calls.
    """

    X: NDArray[np.float32]
    y: NDArray[np.int8]
    feature_columns: tuple[str, ...]
    raw_rows: list[dict[str, float | str | int]]


def generate_dataset(*, seed: int, n: int = 2000) -> Dataset:
    """Deterministically generate ``n`` rows.

    Same ``seed`` -> same X, y, raw_rows. The trainer pins this seed
    so model retraining is reproducible run-to-run.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    rng = np.random.default_rng(seed)

    raw_rows: list[dict[str, float | str | int]] = []
    n_cols = len(FEATURE_COLUMNS)
    X = np.zeros((n, n_cols), dtype=np.float32)
    y = np.zeros(n, dtype=np.int8)

    for i in range(n):
        lead_time_days = int(rng.integers(1, 91))  # 1-90 inclusive
        prior_no_show_rate = float(np.clip(rng.beta(1.5, 6.0), 0.0, 1.0))
        is_new_patient = int(rng.integers(0, 2))
        day_of_week = DAYS_OF_WEEK[int(rng.integers(0, len(DAYS_OF_WEEK)))]
        payer = PAYERS[int(rng.integers(0, len(PAYERS)))]
        age_band = AGE_BANDS[int(rng.integers(0, len(AGE_BANDS)))]
        appointment_type = APPOINTMENT_TYPES[int(rng.integers(0, len(APPOINTMENT_TYPES)))]

        row: dict[str, float | str | int] = {
            "lead_time_days": lead_time_days,
            "prior_no_show_rate": prior_no_show_rate,
            "is_new_patient": is_new_patient,
            "day_of_week": day_of_week,
            "payer": payer,
            "age_band": age_band,
            "appointment_type": appointment_type,
        }
        raw_rows.append(row)

        # Encode features.
        X[i] = encode_row(row)

        # Generative label model: logistic in a few hand-picked drivers.
        # Coefficients keep prior_no_show_rate dominant — that's the
        # signal real PMS no-show models lean on hardest.
        logit = (
            -2.4
            + 4.5 * prior_no_show_rate
            + 0.018 * lead_time_days
            + 0.25 * is_new_patient
            + (0.35 if payer == "self_pay" else 0.0)
            + (0.20 if day_of_week in {"mon", "fri"} else 0.0)
            + (0.15 if appointment_type == "new_patient" else 0.0)
            + float(rng.normal(0.0, 0.6))
        )
        p = 1.0 / (1.0 + np.exp(-logit))
        y[i] = int(rng.random() < p)

    return Dataset(
        X=X,
        y=y,
        feature_columns=FEATURE_COLUMNS,
        raw_rows=raw_rows,
    )


def encode_row(row: dict[str, float | str | int]) -> NDArray[np.float32]:
    """Encode one raw row into the post-one-hot float vector.

    Used both by the dataset generator and by the predictor when
    scoring a single appointment dict at serving time. Unknown
    categorical levels raise — we'd rather fail loudly than silently
    drop signal.
    """
    vec = np.zeros(len(FEATURE_COLUMNS), dtype=np.float32)
    idx = {c: i for i, c in enumerate(FEATURE_COLUMNS)}

    vec[idx["lead_time_days"]] = float(row["lead_time_days"])
    vec[idx["prior_no_show_rate"]] = float(row["prior_no_show_rate"])
    vec[idx["is_new_patient"]] = float(row["is_new_patient"])

    for col_prefix, key, levels in (
        ("day_of_week", "day_of_week", DAYS_OF_WEEK),
        ("payer", "payer", PAYERS),
        ("age_band", "age_band", AGE_BANDS),
        ("appointment_type", "appointment_type", APPOINTMENT_TYPES),
    ):
        level = str(row[key])
        if level not in levels:
            raise ValueError(f"unknown {key}={level!r}; expected one of {levels}")
        vec[idx[f"{col_prefix}={level}"]] = 1.0

    return vec
