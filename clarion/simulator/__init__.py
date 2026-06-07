"""Synthetic patient persona generator + simulation harness."""

from clarion.simulator.generator import generate
from clarion.simulator.templates import (
    DISTRIBUTION,
    OPHTHALMOLOGY,
    ORTHOPEDICS,
    TEMPLATES,
    CustomerTemplate,
)

__all__ = [
    "DISTRIBUTION",
    "OPHTHALMOLOGY",
    "ORTHOPEDICS",
    "TEMPLATES",
    "CustomerTemplate",
    "generate",
]
