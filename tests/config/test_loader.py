"""Integration tests for ``load_customer``.

These exercise the real YAML files in ``configs/`` and tmp-path fixtures for
the error cases. Together with ``test_schema.py`` they enforce Phase 2's
acceptance criterion: *the application can boot entirely from config, with
no hardcoded customer logic.*
"""

from __future__ import annotations

from pathlib import Path

import pytest
from clarion.config.loader import (
    CustomerConfigError,
    CustomerNotFoundError,
    load_customer,
)
from clarion.config.settings import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


def _real_settings(customer: str) -> Settings:
    return Settings(customer=customer, config_dir=CONFIGS_DIR, data_dir=DATA_DIR)


# ---------- real customer configs round-trip ----------


def test_loads_ophthalmology_from_real_yaml() -> None:
    cfg = load_customer(settings=_real_settings("ophthalmology"))
    assert cfg.customer_id == "ophthalmology"
    assert cfg.display_name == "North Shore Eye Associates"
    assert "Cataract Pre-Op Consult" in cfg.specialties
    assert "cancel_appointment" in cfg.enabled_tools
    assert cfg.languages == ["en", "es"]
    # rules_path was relative in the yaml; loader resolves it absolute.
    assert cfg.rules_path.is_absolute()
    assert cfg.rules_path.name == "ophthalmology"


def test_loads_orthopedics_from_real_yaml() -> None:
    cfg = load_customer(settings=_real_settings("orthopedics"))
    assert cfg.customer_id == "orthopedics"
    # Per-customer divergence: orthopedics disables cancel_appointment.
    assert "cancel_appointment" not in cfg.enabled_tools
    assert cfg.escalation.max_clarifications == 4
    assert cfg.languages == ["en"]


def test_default_customer_comes_from_settings() -> None:
    # When no explicit name is passed, loader uses settings.customer.
    cfg = load_customer(settings=_real_settings("orthopedics"))
    assert cfg.customer_id == "orthopedics"


def test_explicit_name_overrides_settings() -> None:
    cfg = load_customer("ophthalmology", settings=_real_settings("orthopedics"))
    assert cfg.customer_id == "ophthalmology"


# ---------- error paths ----------


def test_missing_customer_raises_with_available_listed(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "demo_a.yaml").write_text("customer_id: demo_a\n")
    (cfg_dir / "demo_b.yaml").write_text("customer_id: demo_b\n")
    s = Settings(customer="ghost", config_dir=cfg_dir, data_dir=tmp_path)

    with pytest.raises(CustomerNotFoundError) as exc:
        load_customer(settings=s)
    msg = str(exc.value)
    assert "ghost" in msg
    assert "demo_a" in msg and "demo_b" in msg  # actionable error


def test_yaml_must_be_a_mapping(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "broken.yaml").write_text("- just\n- a\n- list\n")
    s = Settings(customer="broken", config_dir=cfg_dir, data_dir=tmp_path)

    with pytest.raises(CustomerConfigError, match="mapping"):
        load_customer(settings=s)


def test_invalid_schema_raises_customer_config_error(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "bad.yaml").write_text(
        "customer_id: bad\n"
        'display_name: "Bad"\n'
        'specialties: ["X"]\n'
        'enabled_tools: ["search_slots"]\n'
        'rules_path: "rules/bad"\n'
        'agent_persona: "p"\n'
        "escalation:\n"
        "  low_confidence: 5.0\n"  # out of range
    )
    s = Settings(customer="bad", config_dir=cfg_dir, data_dir=tmp_path)

    with pytest.raises(CustomerConfigError) as exc:
        load_customer(settings=s)
    assert "low_confidence" in str(exc.value)


def test_customer_id_defaults_to_filename_stem(tmp_path: Path) -> None:
    cfg_dir = tmp_path / "configs"
    cfg_dir.mkdir()
    (cfg_dir / "inferred.yaml").write_text(
        'display_name: "Inferred"\n'
        'specialties: ["X"]\n'
        'enabled_tools: ["search_slots"]\n'
        'rules_path: "rules/inferred"\n'
        'agent_persona: "p"\n'
    )
    s = Settings(customer="inferred", config_dir=cfg_dir, data_dir=tmp_path)
    cfg = load_customer(settings=s)
    assert cfg.customer_id == "inferred"
