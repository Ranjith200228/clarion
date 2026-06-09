"""Text-based frustration proxy for the escalation engine.

Phase 11 ships a regex-based detector that scores how frustrated the
patient sounds in their *most recent* messages. Phase 18 will fuse this
with a CNN-LSTM voice-emotion signal; the scorer in
``clarion.sentinel.escalation`` will average the two.

Why patterns and not an LLM? Three reasons:

1. **Zero latency on the hot path.** The escalation scorer runs on every
   turn; we don't want a second LLM round-trip just to detect "I already
   told you that."
2. **Auditable.** Every match has a regex you can read.
3. **The LLM-as-judge already covers the subtle stuff** (rule conflicts,
   hallucinations). Frustration is mostly surface-level lexical cues.

The score is in [0, 1] — clamped from a hit-count / saturation function
so multiple weak signals can still cross threshold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Patterns curated from the practice rule corpora's "verbally aggressive"
# and "in obvious distress" callouts in 06_emergencies_and_escalation.md
# of both customers, plus typical irate-caller cues.
_FRUSTRATION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # "I already told you" family.
        r"\bi\s+already\s+(?:told|said|explained)\b",
        r"\b(?:as\s+)?i\s+(?:said|told\s+you)\s+(?:before|already)\b",
        r"\b(?:you'?re|you\s+are)\s+(?:not\s+)?listening\b",
        # Repetition / impatience cues.
        r"\bjust\s+(?:answer|tell|do)\b",
        r"\b(?:why\s+)?(?:can'?t|won'?t)\s+you\b",
        r"\bnot\s+(?:what|that'?s)\s+i\s+(?:asked|said|want)\b",
        # Demand for a human.
        r"\b(?:let|put)\s+me\s+(?:speak|talk|through)\s+to\s+(?:a\s+)?"
        r"(?:human|person|manager|supervisor|someone)\b",
        r"\bput\s+me\s+through\s+to\s+(?:a\s+)?"
        r"(?:human|person|manager|supervisor|someone)\b",
        r"\bi\s+(?:want|need)\s+to\s+(?:speak|talk)\s+to\s+(?:a\s+)?"
        r"(?:human|manager|supervisor|person|someone)\b",
        # Anger / profanity-light markers.
        r"\bthis\s+is\s+(?:ridiculous|insane|unacceptable|absurd)\b",
        r"\bare\s+you\s+(?:kidding|joking|serious)\b",
        # Frustrated commands.
        r"\bstop\s+(?:asking|saying|telling)\b",
        r"\bforget\s+it\b",
        r"\bnever\s+mind\b",
    )
]


@dataclass(frozen=True)
class FrustrationResult:
    """The 0-1 frustration score for one message + which patterns hit.

    ``patterns`` is the list of matched pattern strings so an operator
    auditing a high score can see exactly which cue drove it.
    """

    score: float
    patterns: list[str]
    has_shouting: bool  # all-caps stretch >= 8 chars
    excessive_punctuation: bool  # >= 3 consecutive ? or !


def detect_frustration(message: str) -> FrustrationResult:
    """Score one user message for frustration cues.

    Three classes of signal contribute, each adds to the raw count:

    * Regex pattern hits (one each).
    * Shouting — any all-caps run of 8+ alpha chars.
    * Excessive punctuation — three or more consecutive ! or ?.

    Final score saturates as ``1 - exp(-hits / 2)`` so two strong hits
    already cross 0.6 (i.e. above the default 0.5 escalation threshold)
    but a single weak hit stays below.
    """
    hits: list[str] = []
    for pat in _FRUSTRATION_PATTERNS:
        if pat.search(message):
            hits.append(pat.pattern)

    has_shouting = _has_shouting(message)
    excessive_punct = bool(re.search(r"[!?]{3,}", message))

    raw_count = len(hits) + (1 if has_shouting else 0) + (1 if excessive_punct else 0)
    score = _saturate(raw_count)

    return FrustrationResult(
        score=score,
        patterns=hits,
        has_shouting=has_shouting,
        excessive_punctuation=excessive_punct,
    )


def detect_frustration_over_turns(user_messages: list[str]) -> FrustrationResult:
    """Aggregate frustration across multiple user turns.

    A single conversation accumulates frustration — if the patient said
    "I already told you" three times across three turns, the conversation
    should score higher than any single turn. We sum the per-message hit
    counts (not the saturated scores) and re-saturate once at the end.
    """
    if not user_messages:
        return FrustrationResult(
            score=0.0, patterns=[], has_shouting=False, excessive_punctuation=False
        )

    all_patterns: list[str] = []
    shout = False
    punct = False
    raw_count = 0
    for msg in user_messages:
        per = detect_frustration(msg)
        all_patterns.extend(per.patterns)
        shout = shout or per.has_shouting
        punct = punct or per.excessive_punctuation
        # Re-derive raw count contribution from each message.
        raw_count += (
            len(per.patterns)
            + (1 if per.has_shouting else 0)
            + (1 if per.excessive_punctuation else 0)
        )

    return FrustrationResult(
        score=_saturate(raw_count),
        patterns=all_patterns,
        has_shouting=shout,
        excessive_punctuation=punct,
    )


# ---------- helpers ----------


# A single 8+ char uppercase run OR two 2+ char uppercase words in a row
# both count as shouting. The first catches STOPCALLINGME; the second
# catches "PLEASE LISTEN TO ME" without flagging legitimate single
# abbreviations like USA or ER.
_SHOUT_RUN_RE = re.compile(r"[A-Z]{8,}")
_SHOUT_MULTI_RE = re.compile(r"\b[A-Z]{2,}\b(?:\s+\b[A-Z]{2,}\b){1,}")


def _has_shouting(message: str) -> bool:
    return bool(_SHOUT_RUN_RE.search(message) or _SHOUT_MULTI_RE.search(message))


def _saturate(raw: int) -> float:
    """Saturate raw hit count to [0, 1].

    Uses 1 - exp(-x/2):
      0 hits -> 0.000
      1 hit  -> 0.393
      2 hits -> 0.632   (above default threshold)
      3 hits -> 0.777
      5 hits -> 0.918
    """
    import math

    if raw <= 0:
        return 0.0
    return round(1.0 - math.exp(-raw / 2.0), 4)
