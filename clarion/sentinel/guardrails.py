"""Pre-LLM guardrails — emergency detection and clinical advice refusal.

These run BEFORE the ReAct loop sees a user message. If a guardrail
fires, the loop is skipped: we return a canned, customer-appropriate
response and (for emergencies) file an urgent PMS task. The LLM never
sees the unsafe prompt at all, which means the worst that can happen on
an emergency message is a regex miss — not an LLM hallucinating clinical
advice.

The trigger lists are curated from the rules markdown in
``data/rules/<customer>/06_emergencies_and_escalation.md`` and the
"never give clinical advice" lines from each practice. They're plain
patterns so they're auditable; an LLM-judge layer can vet ambiguous
cases in Phase 10.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

GuardrailKind = Literal["emergency", "clinical_advice", "safe"]


# ---------- emergency patterns ----------
#
# Designed to catch the *patient's words*, not a paramedic's vocabulary —
# someone in distress says "I can't see" or "my eye is bleeding", not
# "ocular trauma with hyphema". Use word-boundary matching so substrings
# inside legitimate words ("strong" containing "stro") don't false-positive.

_EMERGENCY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # General medical emergency markers.
        r"\b(?:call|called|calling)\s+9-?1-?1\b",
        r"\bemergency\s+room\b",
        r"\b(?:having|might be having)\s+(?:a\s+)?stroke\b",
        r"\bheart\s+attack\b",
        r"\bcan'?t\s+breathe\b",
        r"\bunconscious\b",
        r"\bpassed\s+out\b",
        r"\blost\s+consciousness\b",
        # Vision / ophthalmology.
        r"\bsudden(?:ly)?\s+(?:lost|losing)\s+(?:my\s+)?(?:sight|vision)\b",
        r"\b(?:vision|sight)\s+(?:loss|went\s+(?:black|dark))\b",
        r"\bcan'?t\s+see\b",
        r"\beye\s+(?:trauma|injury|bleeding)\b",
        r"\bchemical\s+(?:splash|burn)\s+(?:in|to)\s+(?:my\s+)?eye\b",
        # Orthopedic.
        r"\bcompound\s+fracture\b",
        r"\bbone\s+(?:is\s+)?sticking\s+out\b",
        r"\bcan'?t\s+(?:feel|move)\s+(?:my\s+)?(?:leg|arm|foot|hand)\b",
        r"\b(?:limb|leg|arm|foot|hand)\s+(?:is\s+)?(?:cold|blue|numb)\b",
        r"\bhead\s+injury\b",
        # Cauda equina warning (orthopedic emergency). "lost (control of my)
        # bladder" with optional words between is the practical phrasing.
        r"\b(?:lost|losing|can'?t\s+control)(?:\s+(?:control|all|of|my))*\s+(?:bowel|bladder)\b",
        # Patient-volunteered framing.
        r"\bthis\s+is\s+an?\s+emergency\b",
        r"\b(?:severe|excruciating)\s+pain\b",
    )
]


# ---------- clinical advice patterns ----------
#
# These match requests asking the AGENT for medical judgment. They cover
# common phrasings around medication safety, symptom interpretation, and
# "should I do X" questions where X needs a clinician.

_CLINICAL_ADVICE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bshould\s+i\s+(?:take|stop|skip|increase|decrease|double|change)\b",
        r"\bis\s+(?:it|this)\s+(?:safe|ok|normal|dangerous)\s+(?:to|if|for)\b",
        r"\bcan\s+i\s+take\s+\w+\s+(?:with|while)\b",
        r"\bwhat\s+(?:dose|dosage)\s+(?:of|should)\b",
        r"\bdiagnose\s+(?:me|my)\b",
        r"\bis\s+this\s+(?:cancer|serious|dangerous)\b",
        r"\bdo\s+i\s+(?:have|need)\s+\w+\b(?:.{0,40})\?$",  # "do i have glaucoma?"
        r"\bshould\s+i\s+be\s+worried\s+about\b",
        # Drug interaction / dose questions.
        r"\b(?:interact|interaction)\s+with\s+\w+",
        r"\bhow\s+(?:much|many)\s+ibuprofen\b",
        r"\bdouble\s+(?:my|the)\s+dose\b",
        # Refill requests (per the rule corpora — no refill authority).
        r"\b(?:refill|prescription)\s+(?:my|for)\b",
        r"\brefill\s+(?:my\s+)?(?:medication|drops|prescription)\b",
    )
]


@dataclass(frozen=True)
class GuardrailHit:
    """One guardrail decision about one user message."""

    kind: GuardrailKind
    matched_pattern: str | None = None
    reason: str = ""

    @property
    def fired(self) -> bool:
        return self.kind != "safe"


def detect_emergency(text: str) -> GuardrailHit:
    """Scan ``text`` for patient-described medical emergencies."""
    for pat in _EMERGENCY_PATTERNS:
        if pat.search(text):
            return GuardrailHit(
                kind="emergency",
                matched_pattern=pat.pattern,
                reason=(
                    "Patient described a medical emergency — advise 911 / ED "
                    "and file an urgent PMS task. Do NOT proceed with booking."
                ),
            )
    return GuardrailHit(kind="safe")


def detect_clinical_advice_request(text: str) -> GuardrailHit:
    """Scan ``text`` for requests asking the agent for medical judgment."""
    for pat in _CLINICAL_ADVICE_PATTERNS:
        if pat.search(text):
            return GuardrailHit(
                kind="clinical_advice",
                matched_pattern=pat.pattern,
                reason=(
                    "Patient asked for clinical judgment — refuse politely "
                    "and offer to file a task or schedule a visit."
                ),
            )
    return GuardrailHit(kind="safe")
