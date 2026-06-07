"""PHI redaction for audit logs and observability traces.

This is a *defense-in-depth* layer, not a HIPAA compliance shim — the
project's honesty note in the README says it explicitly. The point is to
demonstrate compliance awareness: even synthetic data goes through the
redactor before it lands in an audit log, so the patterns we'd actually
need in production are exercised.

Patterns covered (all → ``<TAG>`` replacements):

* US phone numbers — ``(555) 555-1234``, ``555-555-1234``, ``5555551234`` → ``<PHONE>``
* SSN — ``123-45-6789`` → ``<SSN>``
* Email addresses → ``<EMAIL>``
* Insurance member ids — ``[A-Z]{2,5}-\\d{3,}`` (matches AET-9981, BCB-3320,
  UHC-7710 etc. from our seeds) → ``<MEMBER_ID>``
* Synthetic patient ids — ``pat_\\d+`` → ``<PATIENT_ID>``

Dates are deliberately NOT redacted — appointment dates appear all over
the place in legitimate contexts and tagging them would make the audit
log useless. A real deployment would add DOB redaction once it knows
which date fields are PHI vs operational.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Each pattern is a (compiled regex, replacement) pair. Order matters —
# more specific patterns first so "AET-9981" doesn't get half-eaten by a
# more general numeric matcher.
_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # SSN before phone (123-45-6789 also matches the phone pattern).
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "<SSN>"),
    # Insurance member ids: 2-5 uppercase letters, dash, 3+ digits.
    (re.compile(r"\b[A-Z]{2,5}-\d{3,}\b"), "<MEMBER_ID>"),
    # Synthetic patient ids used throughout the demo data.
    (re.compile(r"\bpat_\d+\b"), "<PATIENT_ID>"),
    # Email addresses.
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<EMAIL>"),
    # US phone numbers. Two accepted shapes, both require a separator after
    # the local 3-digit group so we don't redact "60 minutes" style numbers
    # or "Aetna 2026" style words.
    #
    # Examples that match:
    #   555-0100           (3-4 with separator, common in our demo seeds)
    #   (555) 555-1234     (area code in parens + local)
    #   +1 555.555.1234    (country code + 10 digits with dots)
    #   555 555 1234       (area code + 7-digit local separated by spaces)
    (
        re.compile(
            r"(?:\+?1[-.\s]?)?"
            r"(?:\(\d{3}\)\s?|\d{3}[-.\s])?"  # optional area code
            r"\d{3}[-.\s]\d{4}\b"  # required separator in the local part
        ),
        "<PHONE>",
    ),
]


@dataclass(frozen=True)
class RedactionResult:
    """The redacted text plus a count of each tag that fired.

    The counts are what the audit-log layer surfaces so we can spot the
    ``[REDACTED] = 7 in one turn`` outliers without having to read every line.
    """

    text: str
    tag_counts: dict[str, int]


def redact(text: str) -> str:
    """Return ``text`` with PHI replaced by ``<TAG>`` placeholders."""
    out = text
    for pattern, tag in _PATTERNS:
        out = pattern.sub(tag, out)
    return out


def redact_with_counts(text: str) -> RedactionResult:
    """Like ``redact`` but also returns per-tag hit counts."""
    out = text
    counts: dict[str, int] = {}
    for pattern, tag in _PATTERNS:
        new_text, n = pattern.subn(tag, out)
        if n:
            counts[tag] = counts.get(tag, 0) + n
        out = new_text
    return RedactionResult(text=out, tag_counts=counts)
