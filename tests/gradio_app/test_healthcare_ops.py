"""Tests for the Healthcare Operations view: data_sources rollup +
view HTML.

Coverage:
- Provider heat map reads providers + availability from SQLite,
  computes per-day utilization, and bands the row status correctly.
- No-show distribution reads existing predictions.jsonl when
  present and falls back to the M3 synthetic generator otherwise.
- PMS tasks parse from task.json files, sorted urgent-first.
- Eligibility bucket counts from SQLite group BY.
- View renders the four panels + handles empty-state inputs.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from pathlib import Path

import pytest

from gradio_app import data_sources
from gradio_app.data_sources import (
    EligibilitySummary,
    HealthcareOpsSnapshot,
    NoShowRiskBucket,
    PmsTaskRow,
    ProviderUtilization,
)
from gradio_app.views import healthcare_ops as view

# ---------- SQLite fixture helpers ----------


_SCHEMA = """
CREATE TABLE providers (
    provider_id TEXT PRIMARY KEY,
    full_name TEXT,
    specialties TEXT,
    location TEXT,
    accepts_new_patients INTEGER
);
CREATE TABLE availability (
    slot_id TEXT PRIMARY KEY,
    provider_id TEXT,
    appointment_type TEXT,
    slot_date TEXT,
    start_time TEXT,
    duration_minutes INTEGER,
    is_booked INTEGER
);
CREATE TABLE eligibility (
    patient_id TEXT PRIMARY KEY,
    payer TEXT,
    member_id TEXT,
    status TEXT,
    plan_name TEXT,
    effective_date TEXT,
    termination_date TEXT
);
"""


def _make_sqlite(
    base: Path,
    customer_id: str,
    *,
    providers: list[tuple[str, str]],
    slots: list[tuple[str, str, str, int]],
    eligibility: list[tuple[str, str, str, str]],
) -> Path:
    """Seed a per-tenant structured.sqlite3 file.

    slots tuples: (slot_id, provider_id, slot_date, is_booked)
    eligibility tuples: (patient_id, payer, member_id, status)
    """
    customer_dir = base / customer_id
    customer_dir.mkdir(parents=True, exist_ok=True)
    path = customer_dir / "structured.sqlite3"
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_SCHEMA)
        conn.executemany(
            "INSERT INTO providers VALUES (?, ?, '[]', 'X', 1)",
            providers,
        )
        conn.executemany(
            "INSERT INTO availability VALUES (?, ?, 'consult', ?, '09:00:00', 30, ?)",
            slots,
        )
        conn.executemany(
            "INSERT INTO eligibility VALUES (?, ?, ?, ?, 'plan', '2026-01-01', '2026-12-31')",
            eligibility,
        )
        conn.commit()
    finally:
        conn.close()
    return path


# ---------- M1 + M3 fixture helpers ----------


def _write_pms_task(
    base: Path, customer_id: str, conv_id: str, payload: dict
) -> None:
    conv_dir = base / customer_id / "pms_writeback" / conv_id
    conv_dir.mkdir(parents=True, exist_ok=True)
    (conv_dir / "task.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_predictions(base: Path, customer_id: str, records: list[dict]) -> None:
    out_dir = base / customer_id / "no_show_prediction"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "predictions.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


@pytest.fixture
def hc_data_dir(tmp_path: Path) -> Path:
    """One tenant with providers, slots, eligibility, PMS tasks."""
    today = date.today()
    days = [today + timedelta(days=i) for i in range(14)]
    # Provider A: most slots booked (high utilization).
    # Provider B: mostly empty.
    slots = []
    # A — 14 slots, 12 booked.
    for i, d in enumerate(days):
        slots.append((f"a_{i}", "prov_a", d.isoformat(), 1 if i < 12 else 0))
    # B — 14 slots, 2 booked.
    for i, d in enumerate(days):
        slots.append((f"b_{i}", "prov_b", d.isoformat(), 1 if i < 2 else 0))
    _make_sqlite(
        tmp_path,
        "ophthalmology",
        providers=[("prov_a", "Dr. Active"), ("prov_b", "Dr. Calm")],
        slots=slots,
        eligibility=[
            ("pat_001", "Aetna", "AET-1", "active"),
            ("pat_002", "Cigna", "CIG-1", "active"),
            ("pat_003", "BCBS", "BCB-1", "pending"),
            ("pat_004", "Medicare", "MED-1", "denied"),
        ],
    )
    _write_pms_task(
        tmp_path,
        "ophthalmology",
        "conv_001",
        {
            "task_id": "task_001",
            "subject": "URGENT: Follow up with patient pat_007",
            "priority": "urgent",
            "assignee_group": "triage",
            "patient_id": "pat_007",
            "generated_at": "2026-06-17T10:00:00Z",
        },
    )
    _write_pms_task(
        tmp_path,
        "ophthalmology",
        "conv_002",
        {
            "task_id": "task_002",
            "subject": "Confirm appointment for pat_003",
            "priority": "normal",
            "assignee_group": "front_desk",
            "patient_id": "pat_003",
            "generated_at": "2026-06-17T11:00:00Z",
        },
    )
    return tmp_path


# ---------- build_healthcare_ops ----------


def test_healthcare_ops_empty_without_artifacts(tmp_path: Path) -> None:
    """Empty data dir -> empty providers / pms / eligibility but
    the synthetic no-show distribution still ships so the page
    isn't visually empty."""
    ops = data_sources.build_healthcare_ops("ghost", data_dir=tmp_path)
    assert ops.has_structured is False
    assert ops.providers == []
    assert ops.pms_tasks == []
    assert ops.eligibility == []
    assert ops.no_show_total > 0
    # 3 buckets in the canonical order.
    assert [b.band for b in ops.no_show_buckets] == ["low", "medium", "high"]


def test_provider_heatmap_reads_sqlite(hc_data_dir: Path) -> None:
    ops = data_sources.build_healthcare_ops(
        "ophthalmology", data_dir=hc_data_dir
    )
    assert ops.has_structured is True
    assert [p.provider_id for p in ops.providers] == ["prov_a", "prov_b"]
    a = ops.providers[0]
    # 12 of 14 booked.
    assert a.slots_total == 14
    assert a.slots_booked == 12
    # > 85% triggers the critical band ("over-booked").
    assert a.status == "critical"
    # Per-day list is one value per grid day.
    assert len(a.daily) == 14


def test_provider_heatmap_bands_low_util_as_unknown(hc_data_dir: Path) -> None:
    ops = data_sources.build_healthcare_ops(
        "ophthalmology", data_dir=hc_data_dir
    )
    b = ops.providers[1]
    # 2 of 14 booked = 14%.
    assert b.slots_total == 14
    assert b.slots_booked == 2
    assert b.utilization < 0.30
    # Under-utilised provider lands in unknown so the heat map
    # doesn't paint empty rows as healthy.
    assert b.status == "unknown"


def test_no_show_distribution_prefers_predictions_file_when_present(
    hc_data_dir: Path,
) -> None:
    _write_predictions(
        hc_data_dir,
        "ophthalmology",
        records=[
            {"p_no_show": 0.10, "risk_band": "low"},
            {"p_no_show": 0.40, "risk_band": "medium"},
            {"p_no_show": 0.95, "risk_band": "high"},
            {"p_no_show": 0.55, "risk_band": "high"},
        ],
    )
    ops = data_sources.build_healthcare_ops(
        "ophthalmology", data_dir=hc_data_dir
    )
    by_band = {b.band: b for b in ops.no_show_buckets}
    assert by_band["low"].count == 1
    assert by_band["medium"].count == 1
    assert by_band["high"].count == 2
    assert ops.no_show_total == 4
    assert ops.no_show_mean_risk == pytest.approx((0.10 + 0.40 + 0.95 + 0.55) / 4)


def test_pms_tasks_sorted_urgent_first(hc_data_dir: Path) -> None:
    ops = data_sources.build_healthcare_ops(
        "ophthalmology", data_dir=hc_data_dir
    )
    assert [t.task_id for t in ops.pms_tasks] == ["task_001", "task_002"]
    assert ops.pms_tasks[0].priority == "urgent"
    assert ops.pms_open_count == 2


def test_eligibility_groups_by_status(hc_data_dir: Path) -> None:
    ops = data_sources.build_healthcare_ops(
        "ophthalmology", data_dir=hc_data_dir
    )
    by_status = {e.status: e for e in ops.eligibility}
    assert by_status["active"].count == 2
    assert by_status["pending"].count == 1
    assert by_status["denied"].count == 1
    assert ops.eligibility_total == 4


# ---------- view: build_html ----------


def _snapshot(
    *,
    has_providers: bool = True,
    has_pms: bool = True,
    has_eligibility: bool = True,
) -> HealthcareOpsSnapshot:
    providers = (
        [
            ProviderUtilization(
                provider_id="prov_a",
                provider_name="Dr. Active",
                slots_total=14,
                slots_booked=12,
                utilization=0.857,
                daily=[0.9] * 14,
                status="critical",
            ),
            ProviderUtilization(
                provider_id="prov_b",
                provider_name="Dr. Calm",
                slots_total=14,
                slots_booked=4,
                utilization=0.286,
                daily=[0.3] * 14,
                status="unknown",
            ),
        ]
        if has_providers
        else []
    )
    pms = (
        [
            PmsTaskRow(
                task_id="task_001",
                subject="URGENT: chest pain caller",
                priority="urgent",
                assignee_group="triage",
                patient_id="pat_007",
                created_at="2026-06-17",
            ),
        ]
        if has_pms
        else []
    )
    eligibility = (
        [
            EligibilitySummary(status="active", count=2, fraction=0.5),
            EligibilitySummary(status="pending", count=1, fraction=0.25),
            EligibilitySummary(status="denied", count=1, fraction=0.25),
        ]
        if has_eligibility
        else []
    )
    return HealthcareOpsSnapshot(
        tenant="Ophthalmology",
        has_structured=has_providers,
        providers=providers,
        days_in_grid=14,
        avg_utilization=0.571 if has_providers else 0.0,
        no_show_buckets=[
            NoShowRiskBucket(band="low", count=10, fraction=0.5),
            NoShowRiskBucket(band="medium", count=6, fraction=0.3),
            NoShowRiskBucket(band="high", count=4, fraction=0.2),
        ],
        no_show_total=20,
        no_show_mean_risk=0.22,
        pms_tasks=pms,
        pms_open_count=len(pms),
        eligibility=eligibility,
        eligibility_total=sum(e.count for e in eligibility),
    )


def test_view_headline_strip_contains_four_kpis() -> None:
    html = view.build_html(_snapshot())
    for label in (
        "PROVIDERS",
        "AVG UTILIZATION",
        "MEAN NO-SHOW RISK",
        "OPEN PMS TASKS",
    ):
        assert label in html


def test_view_renders_heatmap_with_one_row_per_provider() -> None:
    html = view.build_html(_snapshot())
    assert "Dr. Active" in html
    assert "Dr. Calm" in html
    # 14 cells per row x 2 providers = at least 28 day cells.
    cell_count = html.count('title="')
    assert cell_count >= 28


def test_view_no_show_bars_use_band_colours() -> None:
    html = view.build_html(_snapshot())
    # The three band labels appear title-cased.
    for label in ("Low", "Medium", "High"):
        assert label in html


def test_view_eligibility_donut_carries_total_in_centre() -> None:
    html = view.build_html(_snapshot())
    assert "<svg" in html
    # 4 total displayed in donut centre.
    assert ">4<" in html


def test_view_pms_table_shows_urgent_priority_chip() -> None:
    html = view.build_html(_snapshot())
    assert "URGENT" in html
    assert "URGENT: chest pain caller" in html
    assert "triage" in html


def test_view_empty_providers_routes_to_no_sqlite_message() -> None:
    html = view.build_html(_snapshot(has_providers=False))
    assert "No structured.sqlite3 on disk yet" in html


def test_view_empty_pms_routes_to_module_hint() -> None:
    html = view.build_html(_snapshot(has_pms=False))
    assert "Module M1 produces these" in html


def test_view_empty_eligibility_routes_to_sqlite_message() -> None:
    html = view.build_html(_snapshot(has_eligibility=False))
    assert "Eligibility coverage reads from SQLite" in html
