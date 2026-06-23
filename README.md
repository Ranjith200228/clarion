<div align="center">

# Clarion

A configurable multi-agent voice and vision platform for healthcare
front-desk operations.

[![Tests](https://img.shields.io/badge/tests-580%20passing-22D3EE?style=flat-square)](#testing)
[![Python](https://img.shields.io/badge/python-3.11%2B-22D3EE?style=flat-square)](pyproject.toml)
[![Ruff](https://img.shields.io/badge/ruff-clean-22D3EE?style=flat-square)](#testing)
[![License](https://img.shields.io/badge/license-MIT-22D3EE?style=flat-square)](LICENSE)
[![HF Space](https://img.shields.io/badge/demo-Hugging%20Face%20Space-FFD21E?style=flat-square)](huggingface/README.md)

[Live demo](huggingface/README.md) &middot; [How it works](#how-it-works) &middot; [Running it locally](#running-it-locally) &middot; [What I learned](#what-i-learned)

</div>

---

Clarion handles the routine work that absorbs a specialty medical
practice's phone queue &mdash; booking, eligibility checks,
rescheduling, payer questions &mdash; and recognises the calls that
should never reach an automated system. A patient describing sudden
vision loss, a chest pain, a request for clinical advice. The agent
talks through a typed tool registry it can't easily talk around, and
a separate trust engine grades every reply after the fact.

Healthcare is the demonstration vertical, not the product. The core
idea is that vertical-specific behaviour belongs in YAML, not in
code branches. Onboarding a new tenant &mdash; orthopedics today,
dermatology or dental tomorrow &mdash; is a one-day config + seed
job that never touches the agent loop. The repo ships with two
fully-configured tenants (ophthalmology, orthopedics) and the
evaluation harness scores both through the same agent code.

I built this because I wanted to know what it would actually take to
ship a customer-facing AI agent I'd trust at the front-desk of a real
clinic &mdash; not the demo bot, the thing that takes the call. The
answer is that the model is the easy part. The hard parts are the
contracts around it: which tools the agent can see, what it's allowed
to say, how you prove after a bad turn what happened. Most of the
code in this repo is the contracts.

---

## Screenshots

| | |
|:--:|:--:|
| ![Sentinel Operations](docs/screenshots/03-sentinel-ops.png) <br/> **Sentinel Operations.** Composite trust gauge + five-signal breakdown across the last 100 turns. The trust engine grades every agent reply independently of the agent. | ![Agent Flow](docs/screenshots/04-agent-flow.png) <br/> **Agent Flow.** Live trace through the LangGraph for a single conversation: router &rarr; booking specialist &rarr; tools &rarr; Sentinel, with the path the agent actually took highlighted. |
| ![Voice Intelligence](docs/screenshots/05-voice-intelligence.png) <br/> **Voice Intelligence.** Emotion distribution donut + frustration trace + per-turn escalation prediction lifted from voice transcripts. | ![Patient 360](docs/screenshots/06-patient-360.png) <br/> **Patient 360.** Roster chips, profile card, care team, insurance, and a downloadable appointment confirmation generated from the conversation. |
| ![Cost & SLO](docs/screenshots/07-cost-slo.png) <br/> **Cost & SLO.** Per-tenant spend, latency budgets, cost share by tenant. The financial control plane. | ![Invoice OCR](docs/screenshots/07-cost-slo-ocr.png) <br/> **Invoice OCR.** Upload a vendor invoice image, gpt-4o-mini Vision lifts every dollar amount into a structured row with a running total. |
| ![Live Agent](docs/screenshots/08-live-agent.png) <br/> **Live Agent.** Chat with the production LangGraph agent. Tool calls, escalation score, and running cost surface after every turn. | ![Voice Agent](docs/screenshots/09-voice-agent.png) <br/> **Voice Agent.** End-to-end voice round-trip. Whisper transcribes, the same agent core decides, OpenAI TTS reads the reply back. |

---

## How it works

```
                 ┌─────────────────────────────────────────┐
                 │     11-tab Gradio dashboard             │
                 │  reads only clarion.schemas             │
                 └─────────────┬───────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
            ▼                                     ▼
 ┌────────────────────┐              ┌────────────────────────┐
 │  FastAPI service   │              │  Evaluation harness    │
 │  /chat /voice/turn │              │  100 scenarios/tenant  │
 │  /cost/extract-…   │              │  locked schema v1.0.0  │
 └─────────┬──────────┘              └───────────┬────────────┘
           │                                     │
           ▼                                     │
 ┌──────────────────────┐                        │
 │  Sentinel            │                        │
 │  guardrails + judge  │◀───────────────────────┘
 │  + escalation scorer │
 └─────────┬────────────┘
           │
           ▼
 ┌──────────────────────┐
 │  LangGraph agent     │
 │  router → specialist │
 │  → supervisor        │
 └─────────┬────────────┘
           │
           ▼
 ┌──────────────────────┐
 │  Foundation          │
 │  YAML config · RAG · │
 │  SQLite · tools      │
 └──────────────────────┘
```

There are three layers worth talking about.

**The dashboard.** A Gradio Blocks app with eleven tabs. Each tab is a
pure HTML render driven by typed dataclasses from
`clarion.schemas`. The UI module never imports
`clarion.evaluation` &mdash; that boundary is enforced by what
resolves at import time, so a new metric is always a backend change
and the dashboard can't accidentally recompute one. If you try, the
module doesn't load.

**The API.** FastAPI mounts three routes: `/chat` for one text turn,
`/voice/turn` for the speech round trip, and `/cost/extract-invoice`
for the vision OCR. A session manager keys agents on `(customer_id,
conversation_id)` so a session can mix voice and text and the
transcript stays coherent. Every request carries a correlation ID
that flows through into the structured JSON logs.

**The agent.** A LangGraph state graph. A classifier (`IntentRouter`)
maps the user's turn to one of five specialists &mdash; Booking,
Eligibility, Info, Cancel, Emergency. Each specialist sees only the
subset of tools its YAML allows; the Info specialist literally can't
see `book_appointment` on its tool list, so a prompt-injection attempt
has nowhere to reach. After each specialist runs, a supervisor node
decides finish, route to a different specialist, or escalate to a
human.

Around all of that sits **Sentinel**, the trust engine. It has three
independent components, each with a deliberate failure mode:

| Component | Output | Failure mode |
|---|---|---|
| Guardrails (emergency / clinical / PHI) | Short-circuits the reply &mdash; the LLM is never called for an emergency turn | Pattern-based; prefers false alarms |
| LLM-as-Judge (booking + hallucination + policy) | Structured verdict per turn | Defensive parsing &mdash; low-confidence on malformed JSON |
| Escalation scorer | 0&ndash;1 score from five weighted signals | Tunable thresholds per tenant YAML |

The judge runs *after* the agent has replied, so even if the agent
got it right, the system grades the answer from outside. That
separation matters when you're trying to convince yourself a
production system is actually safe.

---

## Things I cared about while building it

A few notes on decisions that are easy to miss skimming the code.

**The agent shouldn't be able to do anything it wasn't supposed to.**
The Pydantic `BookAppointmentInput` rejects free-form text in ID
fields. This started as paranoia and turned into a bug fix the same
week: the LLM hallucinated a caller's full name into the `patient_id`
field once, and the row landed in the SQLite store. The Patient 360
view rendered the corrupted record straight back at me. I dropped
the row by hand and added the regex `^[A-Za-z][A-Za-z0-9_-]{0,63}$`
to the schema. The relevant commit
([`a36a72f`](https://github.com/Ranjith200228/clarion/commit/a36a72f))
ships tests that lock in the rejection of the original bad input.

**The same patient-detail validation runs three layers deep.** The
booking specialist's system prompt explicitly asks the agent to read
the caller's name, phone, and email back to them before calling the
tool. The tool schema validates the same three fields independently
with regex patterns. The store persists them into the appointment's
`notes` column as JSON, so the Patient 360 confirmation card renders
the values the caller actually confirmed &mdash; not a name
synthesised from a hash of the `patient_id`. Three layers because
the prompt is a soft hint and the schema is a hard wall, but a real
contract wants both.

**Tools are advertised per specialist.** The Booking specialist sees
`search_slots`, `book_appointment`, `cancel_appointment`,
`check_eligibility`, `create_pms_task`. The Info specialist sees only
`create_pms_task`. The orthopedics tenant doesn't enable
`cancel_appointment` at all &mdash; its YAML omits it, the registry
honours the allowlist strictly, and that customer's agent has never
seen a cancel tool on any call. Multi-tenancy is an allowlist, not
a code branch. Every time I caught myself reaching for `if customer_id
== "..."` the answer was always to add a YAML field instead.

**The evaluation report contract is locked.** Adding a new metric is
additive only &mdash; new optional field on `EvaluationReport.metrics`,
schema version stays `1.0.0`, the locked-schema regression test
passes. The dashboard reads from this contract; it never recomputes
anything. That has saved hours of debugging when a metric definition
changes and three different pieces of code disagree about which
definition is right.

**Vision OCR was a one-day add-on, not a rewrite.** The Cost & SLO
tab has an invoice uploader that posts to `/cost/extract-invoice`,
which calls gpt-4o-mini in JSON mode with a strict extraction prompt.
The response is parsed defensively (strip code fences, coerce number
strings, drop malformed line items) into the same kind of typed
dataclass the rest of the app uses. Total surface: one Python module,
one FastAPI route, one Gradio component, one CSS block. The module
boundary that made this so cheap to add is the same boundary that
keeps Sentinel decoupled from the agent.

---

## Running it locally

```bash
git clone https://github.com/Ranjith200228/clarion.git
cd clarion
poetry install
export OPENAI_API_KEY=sk-...

# Populate evaluation artifacts (synthetic; run once)
poetry run python -m clarion.eval --customer all

# In one terminal: the FastAPI backend
poetry run python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# In another: the dashboard
poetry run python -m gradio_app

# Open http://localhost:7860
```

The dashboard works without the API for everything that reads
artifacts off disk &mdash; only Live Agent, Voice Agent, and Invoice
OCR need the backend running.

For Docker:

```bash
docker compose up    # API + dashboard side-by-side
```

The multi-stage Dockerfile pre-bakes the FAISS indices in the builder
stage, so a fresh container is serving requests in under two seconds.

---

## Tech stack

| Layer | Notes |
|---|---|
| Language | Python 3.11+, mostly statically typed, mypy strict on the agent core |
| Validation | Pydantic v2 at every boundary; `extra="forbid"` on every tool schema so the model can't smuggle unknown fields |
| LLM | OpenAI &mdash; `gpt-4o-mini` for chat and vision, `whisper-1` for STT, `tts-1` for TTS, `text-embedding-3-small` for RAG |
| Orchestration | LangGraph (StateGraph) for the multi-agent backend; single-`Agent` ReAct loop also available, toggled per tenant in YAML |
| Trust engine | Hand-rolled. Guardrails are regex + keyword, judge is LLM-backed with a strict JSON contract, escalation scorer fuses five signals with tunable weights |
| HTTP | FastAPI with custom middleware for correlation IDs and per-(tenant, IP) token-bucket rate limits |
| UI | Gradio 4.44 Blocks + ~1,300 lines of design-token CSS. Eleven tabs, all driven from typed wire models |
| Resilience | Retry with full-jitter backoff, circuit breaker around the LLM client (5 failures &rarr; 30 s open) |
| Storage | Per-tenant SQLite (`structured.sqlite`) for slots/appointments/eligibility/PMS tasks; FAISS for vector RAG |
| ML side | XGBoost no-show classifier with held-out ROC-AUC and top-decile lift folded into the locked report contract |
| Tests | 580 pytest; CI matrix on 3.11 and 3.12; ruff + mypy gates |
| Deploy | Multi-stage Docker, primary target is a Hugging Face Space; manifests also for Cloud Run, Render, Fly.io |

---

## Repository layout

```
clarion/
  agents/             Single-Agent ReAct loop + OpenAI client wrapper
  multiagent/         LangGraph backend: router, specialists, supervisor
  sentinel/           Trust engine — guardrails, judge, escalation, PHI
  schemas/            Pydantic wire models. The contract between layers.
  modules/            Opt-in post-launch capabilities
    invoice_ocr.py        gpt-4o-mini Vision invoice extraction
    no_show_prediction/   XGBoost classifier, persisted to joblib
    pms_writeback/        Conversation → summary.json + task.json
    voice/                Whisper + OpenAI TTS round trip
  pipelines/          Structured store (SQLite) + RAG retriever
  resilience/         Retry, circuit breaker, rate limit
  evaluation/         100-scenario harness, locked report writer
  observability/      Structured JSON logs, correlation IDs, spans
  config/             Settings + per-tenant YAML loader

api/
  app.py              FastAPI factory
  routes/             /chat /voice/turn /cost/extract-invoice /health
  middleware/         Correlation IDs + rate limiter
  sessions.py         Per-(tenant, conversation) session manager

gradio_app/
  app.py              11-tab Blocks shell + customer switcher
  views/              One file per tab, pure HTML render functions
  components.py       Shared visual primitives (KPI tile, donut, …)
  data_sources.py     Typed roll-ups consumed by the views
  tab_*.py            Stateful tabs (live agent, voice agent, cost OCR)
  theme.py / style.css   Design tokens and primitive CSS

configs/              Per-tenant YAML
data/                 Per-tenant artifacts (gitignored — re-generated by harness)
tests/                580 pytest tests (unit + integration + e2e)
docs/                 Discovery doc, dev/deploy guides, security review
loadtest/             Locust profile + in-process p95 SLA test
huggingface/          HF Spaces deployment manifest
```

---

## Testing

```bash
poetry run pytest                              # the full 580
poetry run ruff check clarion gradio_app api tests
poetry run mypy --strict clarion api
poetry run pytest -m loadtest                  # opt-in p95 SLA budget
```

The 580 tests include:
- The e2e ophthalmology and orthopedics booking flows, driven by a
  scripted `FakeLLM` so CI never needs a real key.
- The boundary regex guards on `BookAppointmentInput` &mdash; six
  cases covering single-word names, "ask me later" phones, "n/a"
  emails, and the international/E.164 formats they should still
  accept.
- A schema-regression test on the locked `EvaluationReport` v1.0.0
  contract. Anything additive is fine; anything that would change
  an existing field's name or type fails the gate.
- An opt-in `loadtest` marker that runs an in-process p95 latency
  budget against a `FakeLLM` so the numbers reflect framework +
  middleware + agent overhead. For real-OpenAI numbers there's also
  a Locust profile in `loadtest/locustfile.py` you can run against
  a deployed instance.

---

## What I learned

A few things that surprised me while building this, in case any of it
is useful to someone making the same calls.

The hardest part was never the model. It was deciding what the
agent's tools *aren't* allowed to do. Pydantic regex guards on ID
fields prevented exactly one production bug (the patient-name-as-id
incident) and probably several I never noticed. The cost was a few
lines of code; the value was the dashboard not silently rendering
garbage.

LangGraph specialists are mostly a vehicle for tool scoping. The
classifier hop adds latency I'd rather not have. What I do want is
that the Info specialist literally can't book an appointment because
it's never been advertised the tool. That's a property of the graph
structure, not the prompt, which means a jailbreak attempt has no
target to aim at.

The independent judge has been worth its cost. It's caught two
real-feeling bugs during eval: a hallucinated specialty
("retinal-vascular consult") on a tenant that doesn't offer it, and
a booking where the agent confirmed a time that conflicted with the
slot it had actually reserved. Both bugs that would have been
plausible to a human reviewer. The judge flagged them because it
runs from outside.

Pre-aggregated UI feeds in the locked report mean the dashboard tabs
are mostly 30-line render functions. Every time I added a new view
my first instinct was "let me just call the metric function". Every
time I resisted, the view ended up smaller and the schema ended up
clearer. The instinct is wrong &mdash; the discipline is right.

---

## Status

Shipped, in this order:

- v1.0.0 (tagged 2026-06-12) &mdash; single-Agent ReAct backend,
  Sentinel trust engine, locked evaluation schema, FastAPI service,
  production hardening, 100-scenario reports for both tenants.
- LangGraph multi-agent backend, opt-in per tenant via `use_multiagent: true`.
- PMS writeback module &mdash; conversation to `summary.json` +
  `task.json` with PHI redaction at the writer.
- No-show prediction (XGBoost) folded into the locked report.
- Voice layer &mdash; Whisper STT + OpenAI TTS round trip.
- Gradio dashboard rewrite to the eleven-tab shell.
- Invoice OCR via gpt-4o-mini Vision.
- Engagement layer (time-of-day greeting, per-tenant accent identity,
  KPI value entrance animation).
- Boundary regex guards on every tool ID field after the patient-id
  hallucination incident.
- Caller-confirmed contact details (name, phone, email) flow from
  the conversation through to the printable confirmation.

What's next:
- Live-mode evaluation with real OpenAI numbers in the report.
- Investigation: a fine-tuned classifier behind `BookingSpecialist` to
  cut prompt size and latency.
- Push from the Hugging Face Space to a custom domain.

---

## Reading deeper

| Document | What's in it |
|---|---|
| [docs/discovery.md](docs/discovery.md) | The pre-build artifact: customer problem, requirements, how the architecture maps onto them |
| [docs/developer_guide.md](docs/developer_guide.md) | Local setup, the six steps for onboarding a new tenant, internal layout |
| [docs/deployment_guide.md](docs/deployment_guide.md) | Container build, secret management, the four hosting targets |
| [docs/security_review.md](docs/security_review.md) | STRIDE-shaped audit. The gaps section is the honest part |
| [reports/v1.0.0/](reports/v1.0.0/) | Locked-schema evaluation reports for both tenants |
| [CHANGELOG.md](CHANGELOG.md) | What shipped, with pointers into the code |

---

[MIT](LICENSE) &copy; 2026 Ranjith Maddirala.

Happy to walk through any layer of this in detail &mdash;
`ranjithmaddirala24@gmail.com`.
