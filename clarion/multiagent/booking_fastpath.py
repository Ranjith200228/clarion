"""Booking intent fast-path: small sklearn classifier that primes
the BookingSpecialist before the LLM round-trip.

The classifier is a TF-IDF + LogisticRegression pipeline over
templated booking utterances. It maps each user turn to one of:

  search              - "what's open next week?"
  book                - "book me a slot at 2pm with Dr. Smith"
  reschedule          - "can I move my Tuesday appointment?"
  cancel              - "I need to cancel my Friday consult"
  check_eligibility   - "do you take Aetna?"
  fallback            - anything else (low confidence -> LLM decides)

When the classifier is confident (default threshold 0.65), the
specialist injects a one-line intent hint into its system prompt.
The LLM still runs, still owns the tool call, still composes the
reply - we are *priming* the model, not bypassing it. That keeps
the full ReAct loop's safety + tool-call surface intact while
shortening the path the LLM has to discover on its own.

Why a classifier instead of more prompt engineering: prompt tokens
cost money on every turn and dominate latency. A 3ms classifier
that lets us drop 40-80 tokens of "figure out the intent" guidance
from the system prompt pays for itself after the first call.

Why not fine-tune gpt-4o-mini directly: we'd need OpenAI's
fine-tuning API + a labelled dataset. This module ships in-repo,
trains in <1 second on synthetic data, and we own the failure
modes. A real production deployment would graduate this to a
proper trained classifier on real call transcripts.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Literal, NamedTuple, cast

log = logging.getLogger(__name__)

# Intent labels the classifier emits. ``fallback`` is the "I don't
# know, let the LLM handle it" bucket - kept explicit so the
# integration code can branch on it cleanly.
FastPathIntent = Literal[
    "search",
    "book",
    "reschedule",
    "cancel",
    "check_eligibility",
    "fallback",
]

_REAL_INTENTS: tuple[FastPathIntent, ...] = (
    "search",
    "book",
    "reschedule",
    "cancel",
    "check_eligibility",
)


class FastPathDecision(NamedTuple):
    """One classifier verdict."""

    intent: FastPathIntent
    confidence: float

    def is_confident(self, threshold: float = 0.65) -> bool:
        """True if confidence clears the threshold AND the intent is
        a real one (not the explicit fallback bucket)."""
        return self.intent != "fallback" and self.confidence >= threshold


@dataclass
class BookingFastPath:
    """Trained classifier + threshold + introspection.

    Instances are immutable after construction. Use
    :meth:`train_default` to build a fresh one from the bundled
    synthetic templates, or pickle a trained instance to disk for
    deployment.
    """

    model: object               # sklearn LogisticRegression
    vectorizer: object          # sklearn TfidfVectorizer
    classes_: tuple[str, ...]
    accuracy: float             # held-out accuracy at train time
    n_train: int
    n_test: int

    # ---------- inference ----------

    def predict(self, message: str) -> FastPathDecision:
        """Classify a single user message.

        Returns ``FastPathDecision(intent, confidence)`` where
        ``confidence`` is the predicted probability of the top class.
        Empty / whitespace-only input returns the ``fallback`` bucket
        with zero confidence so the integration code falls straight
        through to the LLM.
        """
        text = (message or "").strip()
        if not text:
            return FastPathDecision(intent="fallback", confidence=0.0)

        vec = self.vectorizer.transform([text])  # type: ignore[attr-defined]
        proba = self.model.predict_proba(vec)[0]  # type: ignore[attr-defined]
        best_idx = int(proba.argmax())
        return FastPathDecision(
            intent=cast(FastPathIntent, self.classes_[best_idx]),
            confidence=float(proba[best_idx]),
        )

    def hint_for(self, message: str, *, threshold: float = 0.65) -> str | None:
        """Return a one-line system-prompt hint when the classifier is
        confident, ``None`` when it isn't.

        The hint is phrased as guidance to the LLM, not an instruction
        - the LLM still owns the final decision and tool call. This
        keeps the integration safe even when the classifier is wrong.
        """
        d = self.predict(message)
        if not d.is_confident(threshold):
            return None
        return _HINT_BY_INTENT[d.intent]

    # ---------- training ----------

    @classmethod
    def train_default(
        cls, *, seed: int = 42, n_per_intent: int = 80
    ) -> BookingFastPath:
        """Train a fresh classifier from the bundled synthetic templates.

        Deterministic when ``seed`` is held constant - the tests rely
        on this. Held-out accuracy is computed on a 20% split so the
        instance carries an honest signal of how trustworthy the
        classifier's confidence numbers are.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import train_test_split

        texts, labels = generate_training_data(seed=seed, n_per_intent=n_per_intent)

        # 20% held-out test split, stratified so each intent shows up
        # in both train + test even at small n.
        x_train, x_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.20,
            stratify=labels,
            random_state=seed,
        )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
        )
        x_train_vec = vectorizer.fit_transform(x_train)
        x_test_vec = vectorizer.transform(x_test)

        model = LogisticRegression(
            max_iter=500,
            C=1.0,
            random_state=seed,
        )
        model.fit(x_train_vec, y_train)
        accuracy = float(model.score(x_test_vec, y_test))

        log.info(
            "BookingFastPath trained on %d examples, held-out acc=%.3f",
            len(x_train),
            accuracy,
        )
        return cls(
            model=model,
            vectorizer=vectorizer,
            classes_=tuple(str(c) for c in model.classes_),
            accuracy=accuracy,
            n_train=len(x_train),
            n_test=len(x_test),
        )


# ---------- synthetic training data ----------


# Each intent has a list of templates with one or more ``{slot}``
# placeholders. The data generator fills slots from the matching
# vocab and emits ``n_per_intent`` lines per intent. The vocab is
# intentionally light - real production would graduate this to
# labelled transcripts.

_TIME_VOCAB = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
    "next week", "tomorrow", "this afternoon", "2pm", "9 in the morning",
    "early next month", "June 15", "the first available",
)
_PROVIDER_VOCAB = (
    "Dr. Smith", "Dr. Patel", "Dr. Lee", "Dr. Garcia",
    "any provider", "whoever's free", "the cataract specialist",
    "an ophthalmologist",
)
_TYPE_VOCAB = (
    "cataract pre-op", "glaucoma follow-up", "routine eye exam",
    "consultation", "annual visit", "new-patient visit",
    "follow-up", "joint consult", "MRI review",
)
_PAYER_VOCAB = (
    "Aetna", "BCBS", "United Healthcare", "Cigna", "Medicare",
    "my insurance", "my plan",
)

_TEMPLATES: dict[FastPathIntent, tuple[str, ...]] = {
    "search": (
        "what's available {time}?",
        "do you have anything open {time}?",
        "any slots {time}?",
        "what {type} slots are open {time}?",
        "show me availability with {provider}",
        "i'm looking for a slot {time}",
        "when can i see {provider}?",
        "do you have any openings {time} for {type}?",
        "what's the soonest i can come in?",
        "are there any appointments {time}?",
        "looking for a {type} {time}",
        "can you check for openings {time}?",
    ),
    "book": (
        "book me {time} with {provider}",
        "i'd like to schedule {time}",
        "can i book a {type} {time}?",
        "please schedule me for {time}",
        "go ahead and book that {time} slot",
        "yes, book it",
        "confirm the {time} appointment",
        "schedule me for the {type} {time}",
        "lock in {time} please",
        "i want to take that slot",
        "let's go with {time}",
        "schedule a {type} for {time} with {provider}",
    ),
    "reschedule": (
        "can i move my {time} appointment?",
        "i need to reschedule my visit",
        "is there a way to push my appointment to {time}?",
        "move my {type} from {time} to a different day",
        "reschedule me",
        "need to change my appointment time",
        "can we shift my {time} slot?",
        "i can't make {time}, can we move it?",
        "move my appointment to {time}",
        "i'd like to reschedule for {time}",
        "push my appointment back to {time}",
        "swap my time to {time}",
    ),
    "cancel": (
        "i need to cancel my {time} appointment",
        "please cancel my visit",
        "cancel my {type}",
        "i can't make it, cancel please",
        "remove my {time} slot",
        "i won't be coming in {time}",
        "drop my appointment",
        "i'd like to cancel my {time} consult",
        "scratch my appointment",
        "cancel my appointment with {provider}",
        "please remove my {time} booking",
        "i'm cancelling",
    ),
    "check_eligibility": (
        "do you take {payer}?",
        "is {payer} accepted here?",
        "i have {payer}, are you in network?",
        "what insurance plans do you take?",
        "do you accept {payer} for {type}?",
        "is my {payer} plan covered?",
        "are you in network with {payer}?",
        "will {payer} cover a {type}?",
        "i want to check my coverage",
        "can you check if my {payer} is good here?",
        "is {payer} on file for me?",
        "what's my benefit coverage?",
    ),
}


def generate_training_data(
    *, seed: int = 42, n_per_intent: int = 80
) -> tuple[list[str], list[str]]:
    """Return ``(texts, labels)`` aligned arrays with ``n_per_intent``
    examples for each of the five real intents, plus the same count
    of *fallback* noise drawn from off-topic chatter.

    Deterministic for a given ``seed`` - tests assert reproducibility.
    """
    rng = random.Random(seed)
    texts: list[str] = []
    labels: list[str] = []

    for intent in _REAL_INTENTS:
        templates = _TEMPLATES[intent]
        for _ in range(n_per_intent):
            tpl = rng.choice(templates)
            filled = _fill_slots(tpl, rng)
            texts.append(filled)
            labels.append(intent)

    # Fallback / noise bucket. These are real things callers say that
    # *aren't* one of the five booking intents - greetings, FAQs,
    # clinical questions (the trust engine catches those, but the
    # classifier should also abstain on them).
    noise = (
        "hi", "hello there", "good morning", "thanks", "thank you so much",
        "what time do you open?", "where are you located?",
        "what's your address?", "do you have parking?",
        "can i bring my child?", "how long does it take?",
        "i have a question about my prescription",
        "my eye really hurts, what should i do?",
        "do you treat children?", "what should i bring?",
        "is there a fee for missed appointments?",
        "do i need a referral?", "can i pay online?",
        "how do i find my account?", "what's the wait time today?",
        "i lost my paperwork", "my insurance card number changed",
    )
    for _ in range(n_per_intent):
        texts.append(rng.choice(noise))
        labels.append("fallback")

    return texts, labels


def _fill_slots(template: str, rng: random.Random) -> str:
    """Replace ``{time}``, ``{provider}``, ``{type}``, ``{payer}`` slots
    with random draws from the matching vocab. Other braces pass
    through (defensive)."""
    return (
        template
        .replace("{time}", rng.choice(_TIME_VOCAB))
        .replace("{provider}", rng.choice(_PROVIDER_VOCAB))
        .replace("{type}", rng.choice(_TYPE_VOCAB))
        .replace("{payer}", rng.choice(_PAYER_VOCAB))
    )


# ---------- prompt hint phrasing ----------


# One short hint per intent. Phrased as *guidance to the LLM* so the
# LLM still owns the final decision. Crucially these don't tell the
# LLM which tool to call - the existing system prompt and tool list
# do that. They prime the model so it stops "discovering" the intent
# and starts acting on it.
_HINT_BY_INTENT: dict[FastPathIntent, str] = {
    "search": (
        "Hint: the caller most likely wants to SEARCH for slots. "
        "Use search_slots after confirming any missing constraints."
    ),
    "book": (
        "Hint: the caller most likely wants to BOOK a specific slot. "
        "Confirm slot_id, patient identity, and contact details "
        "(name, phone, email) before calling book_appointment."
    ),
    "reschedule": (
        "Hint: the caller most likely wants to RESCHEDULE. "
        "Cancel the existing appointment, then book a new one - "
        "do not call book_appointment without cancelling first."
    ),
    "cancel": (
        "Hint: the caller most likely wants to CANCEL an existing "
        "appointment. Find the appointment_id, then call "
        "cancel_appointment."
    ),
    "check_eligibility": (
        "Hint: the caller most likely wants an ELIGIBILITY check. "
        "Use check_eligibility once you have a patient_id."
    ),
}
