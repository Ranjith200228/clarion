"""Module M3: No-Show Prediction.

Train a calibrated XGBoost classifier on synthetic appointment data,
then serve per-appointment risk scores at inference time. Outputs:

  <data_dir>/<customer_id>/no_show_prediction/predictions.jsonl
      one NoShowPrediction per scored appointment
  models/no_show_v1.joblib
      persisted booster + NoShowModelMetadata bundle

Public surface (built up across commits):
  generate_dataset, encode_row, FEATURE_COLUMNS  (commit 2)
  train, persist, TrainResult                    (commit 3)
  NoShowPredictor                                (commit 4)
  compute_top_decile_lift                        (commit 5)
"""

from clarion.modules.no_show_prediction.dataset import (
    AGE_BANDS,
    APPOINTMENT_TYPES,
    DAYS_OF_WEEK,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    PAYERS,
    Dataset,
    encode_row,
    generate_dataset,
)

__all__ = [
    "AGE_BANDS",
    "APPOINTMENT_TYPES",
    "DAYS_OF_WEEK",
    "Dataset",
    "FEATURE_COLUMNS",
    "NUMERIC_FEATURES",
    "PAYERS",
    "encode_row",
    "generate_dataset",
]
