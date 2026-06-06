"""Process-wide settings for Clarion.

Everything that varies per deployment (which customer to load, where configs
and data live, log level) is read here. Per-customer business logic lives in
``CustomerConfig`` (see ``clarion.config.schema``), loaded from a YAML file
selected by ``Settings.customer``.

Settings can be overridden via environment variables with the ``CLARION_``
prefix, or via a ``.env`` file at the repo root.

Examples::

    # In .env:
    CLARION_CUSTOMER=orthopedics

    # Or on the command line:
    CLARION_CUSTOMER=orthopedics python -m clarion ...
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Deployment-level settings, sourced from environment + ``.env``."""

    model_config = SettingsConfigDict(
        env_prefix="CLARION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Which customer config to load. Must match a ``<customer>.yaml`` filename
    # in ``config_dir``.
    customer: str = Field(default="ophthalmology", description="Customer config to load")

    # Where customer YAMLs live (one per customer).
    config_dir: Path = Field(default=_REPO_ROOT / "configs")

    # Where data artifacts live (rules corpora, sqlite, personas).
    data_dir: Path = Field(default=_REPO_ROOT / "data")

    # Logging.
    log_level: str = Field(default="INFO")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings (cached).

    Tests that need a fresh read can call ``get_settings.cache_clear()``.
    """
    return Settings()
