"""Seed a customer's structured store from a JSON file.

JSON shape is the same one the demo seeds use::

    {
      "providers":   [ {Provider}, ... ],
      "availability":[ {AvailabilitySlot}, ... ],
      "eligibility": [ {EligibilityRecord}, ... ]
    }

Lookup order for the seed file when only a customer_id is given:

    <data_dir>/seeds/<customer_id>.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from clarion.pipelines.structured.store import StructuredStore
from clarion.schemas import AvailabilitySlot, EligibilityRecord, Provider


@dataclass(frozen=True)
class SeedSummary:
    providers: int
    slots: int
    eligibility: int


def default_seed_path(customer_id: str, data_dir: Path) -> Path:
    return data_dir / "seeds" / f"{customer_id}.json"


def seed_structured(
    customer_id: str,
    *,
    data_dir: Path,
    store: StructuredStore | None = None,
    seed_path: Path | None = None,
) -> SeedSummary:
    """Load seeds into the customer's structured store.

    Idempotent — uses upsert under the hood, so re-seeding overwrites in
    place rather than failing on PK collision.
    """
    seed_path = seed_path or default_seed_path(customer_id, data_dir)
    if not seed_path.is_file():
        raise FileNotFoundError(
            f"No seed file for '{customer_id}' at {seed_path}. " f"Create one or pass --seed."
        )
    payload = json.loads(seed_path.read_text(encoding="utf-8"))

    store = store or StructuredStore.for_customer(customer_id, data_dir)

    for p in payload.get("providers", []):
        store.upsert_provider(Provider(**p))
    for s in payload.get("availability", []):
        store.upsert_slot(AvailabilitySlot(**s))
    for e in payload.get("eligibility", []):
        store.upsert_eligibility(EligibilityRecord(**e))

    return SeedSummary(
        providers=len(payload.get("providers", [])),
        slots=len(payload.get("availability", [])),
        eligibility=len(payload.get("eligibility", [])),
    )
