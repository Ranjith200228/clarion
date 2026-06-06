"""Load a customer YAML into a validated ``CustomerConfig``.

Path resolution rules:

* ``load_customer("ophthalmology")`` reads ``<config_dir>/ophthalmology.yaml``
  where ``config_dir`` comes from ``Settings`` (overridable via
  ``CLARION_CONFIG_DIR``).
* ``rules_path`` inside the YAML may be relative; it is resolved against the
  ``Settings.data_dir`` (overridable via ``CLARION_DATA_DIR``) so configs are
  portable across machines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from clarion.config.schema import CustomerConfig
from clarion.config.settings import Settings, get_settings


class CustomerNotFoundError(FileNotFoundError):
    """Raised when no YAML matches the requested customer name."""


class CustomerConfigError(ValueError):
    """Raised when a YAML exists but fails schema validation."""


def _resolve_rules_path(raw: str | Path, data_dir: Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (data_dir / p).resolve()


def load_customer(
    customer: str | None = None,
    *,
    settings: Settings | None = None,
) -> CustomerConfig:
    """Load and validate one customer config.

    Args:
        customer: Customer id (YAML filename stem). Defaults to
            ``settings.customer`` so the app boots from env.
        settings: Override settings (mainly for tests).

    Raises:
        CustomerNotFoundError: No ``<customer>.yaml`` in the config dir.
        CustomerConfigError: YAML exists but fails schema validation.
    """
    settings = settings or get_settings()
    name = customer or settings.customer

    yaml_path = settings.config_dir / f"{name}.yaml"
    if not yaml_path.is_file():
        raise CustomerNotFoundError(
            f"No customer config found for '{name}' at {yaml_path}. "
            f"Available: {_available_customers(settings.config_dir)}"
        )

    raw: dict[str, Any] = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise CustomerConfigError(
            f"{yaml_path} must contain a YAML mapping at the top level, "
            f"got {type(raw).__name__}."
        )

    # Resolve rules_path against data_dir before validation so the schema
    # sees an absolute path.
    if "rules_path" in raw:
        raw["rules_path"] = _resolve_rules_path(raw["rules_path"], settings.data_dir)

    # Fill in customer_id from filename if YAML omitted it — convention over
    # configuration, matches the "drop a file in configs/" story.
    raw.setdefault("customer_id", name)

    try:
        return CustomerConfig(**raw)
    except ValidationError as e:
        raise CustomerConfigError(f"Invalid customer config at {yaml_path}:\n{e}") from e


def _available_customers(config_dir: Path) -> list[str]:
    if not config_dir.is_dir():
        return []
    return sorted(p.stem for p in config_dir.glob("*.yaml"))
