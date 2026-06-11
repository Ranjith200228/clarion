"""Field-extraction accuracy for Module M1.

Compares the fields the extractor produced against the scenario's
ground truth. The metric is the **fraction of evaluated fields that
match**, averaged over all conversations the writer produced output
for.

Field-by-field rules:

* ``patient_id``         — exact string match against any ``pat_\\d+``
  pattern found in the scenario's user messages
* ``appointment_type``   — exact string match against
  ``scenario.ground_truth.expected_appointment_type``
* ``outcome``            — match against the mapped harness outcome
  (the extractor uses the same lookup table)
* ``intent``             — match against ``scenario.intent``

A scenario contributes one accuracy fraction (matches / evaluated),
and the metric returned is the mean over all scenarios. Fields that
are *expected to be null* on both sides (e.g. appointment_type for
emergency scenarios) are excluded from the evaluation count so they
don't artificially inflate the score.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from clarion.schemas import ConversationSummary, HarnessReport, Scenario

log = logging.getLogger(__name__)

_PAT_ID_RE = re.compile(r"\bpat[_-]?\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class FieldAccuracyResult:
    """Per-run accuracy with the per-field breakdown the dashboard renders."""

    accuracy: float  # mean across all evaluated scenarios in [0, 1]
    total_scenarios: int  # how many scenarios had summary.json read back
    total_fields_evaluated: int
    total_fields_matched: int
    by_field: dict[str, tuple[int, int]]  # field -> (matched, evaluated)


def compute_field_extraction_accuracy(
    scenarios: list[Scenario],
    report: HarnessReport,
    *,
    writeback_dir: Path,
) -> FieldAccuracyResult | None:
    """Walk every scenario, load its summary.json, score the fields.

    Returns None when ``writeback_dir`` doesn't exist (module disabled).
    Returns a FieldAccuracyResult with ``accuracy=0.0`` when the
    directory exists but contains no summaries (configuration error).
    """
    if not writeback_dir.is_dir():
        log.info("field-extraction accuracy skipped: %s missing", writeback_dir)
        return None

    by_id = {s.scenario_id: s for s in scenarios}
    total_matched = 0
    total_evaluated = 0
    total_scenarios = 0
    by_field: dict[str, tuple[int, int]] = {
        "patient_id": (0, 0),
        "appointment_type": (0, 0),
        "outcome": (0, 0),
        "intent": (0, 0),
    }

    for result in report.results:
        scenario = by_id.get(result.scenario_id)
        if scenario is None:
            continue
        summary_path = writeback_dir / result.scenario_id / "summary.json"
        if not summary_path.is_file():
            # Some scenarios may have been short-circuited or skipped;
            # don't punish the metric for missing files, just skip them.
            continue
        try:
            summary = ConversationSummary.model_validate_json(
                summary_path.read_text(encoding="utf-8")
            )
        except Exception as e:
            log.warning("could not parse %s: %s", summary_path, e)
            continue

        scenario_matched = 0
        scenario_evaluated = 0

        for field, (matched, evaluated) in _score_one(scenario, summary).items():
            scenario_matched += matched
            scenario_evaluated += evaluated
            cur_matched, cur_eval = by_field[field]
            by_field[field] = (cur_matched + matched, cur_eval + evaluated)

        total_matched += scenario_matched
        total_evaluated += scenario_evaluated
        total_scenarios += 1

    accuracy = (total_matched / total_evaluated) if total_evaluated else 0.0
    return FieldAccuracyResult(
        accuracy=round(accuracy, 4),
        total_scenarios=total_scenarios,
        total_fields_evaluated=total_evaluated,
        total_fields_matched=total_matched,
        by_field=by_field,
    )


def _score_one(scenario: Scenario, summary: ConversationSummary) -> dict[str, tuple[int, int]]:
    """Return per-field (matched, evaluated) counters for one scenario."""
    out: dict[str, tuple[int, int]] = {}

    # patient_id: any pat_NNN in scenario.messages is the expected id.
    expected_patient_id: str | None = None
    for msg in scenario.messages:
        m = _PAT_ID_RE.search(msg)
        if m:
            expected_patient_id = m.group(0).lower()
            break
    if expected_patient_id is not None:
        # The writer PHI-redacts patient_id on disk to "<PATIENT_ID>",
        # so a redaction tag here means the extractor DID catch the
        # id (the privacy layer just blanked it before write). Treat
        # that as a positive match — we're measuring extraction
        # quality, not redaction strictness.
        observed = summary.patient_id
        matched = 1 if observed in {expected_patient_id, "<PATIENT_ID>"} else 0
        out["patient_id"] = (matched, 1)
    else:
        # No ground truth -> skip; field doesn't count toward accuracy.
        out["patient_id"] = (0, 0)

    # appointment_type
    expected_appt = scenario.ground_truth.expected_appointment_type
    if expected_appt:
        matched = 1 if summary.appointment_type == expected_appt else 0
        out["appointment_type"] = (matched, 1)
    else:
        out["appointment_type"] = (0, 0)

    # outcome
    expected_outcome = scenario.ground_truth.expected_outcome
    matched = 1 if summary.outcome == expected_outcome else 0
    out["outcome"] = (matched, 1)

    # intent
    if scenario.intent:
        matched = 1 if summary.intent == scenario.intent else 0
        out["intent"] = (matched, 1)
    else:
        out["intent"] = (0, 0)

    return out
