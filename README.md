---
title: Clarion — Configurable Multi-Agent Voice Automation
emoji: 🛎️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
suggested_storage: small
pinned: false
license: mit
short_description: Multi-tenant healthcare scheduling + Sentinel trust engine
tags:
  - agents
  - rag
  - evaluation
  - observability
  - fastapi
  - gradio
---

<div align="center">

# Clarion

**A production-grade, multi-tenant, multi-agent voice + vision platform for healthcare operations.**

Engineered around three load-bearing principles: typed contracts at every boundary, an independent trust engine that grades the AI from outside, and config-driven multi-tenancy so onboarding a new vertical never touches the agent loop.

[![Tests](https://img.shields.io/badge/tests-705%20passing-22D3EE?style=flat-square)](#evaluation--results)
[![Python](https://img.shields.io/badge/python-3.11%2B-22D3EE?style=flat-square)](pyproject.toml)
[![mypy strict](https://img.shields.io/badge/mypy-strict-22D3EE?style=flat-square)](#evaluation--results)
[![Ruff](https://img.shields.io/badge/ruff-clean-22D3EE?style=flat-square)](#evaluation--results)
[![Schema](https://img.shields.io/badge/eval%20schema-v1.0.0%20locked-22D3EE?style=flat-square)](#evaluation--results)
[![License](https://img.shields.io/badge/license-MIT-22D3EE?style=flat-square)](LICENSE)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-HuggingFace%20Space-FFD21E?style=flat-square)](https://huggingface.co/spaces/Ranjithmaddirala/clarion)

**[Live Demo](https://huggingface.co/spaces/Ranjithmaddirala/clarion)**
&nbsp;&middot;&nbsp;
**[Architecture](#system-architecture)**
&nbsp;&middot;&nbsp;
**[Evaluation & Results](#evaluation--results)**
&nbsp;&middot;&nbsp;
**[Engineering Decisions](#engineering-decisions)**
&nbsp;&middot;&nbsp;
**[Quick Start](#quick-start)**

</div>

---

## At a Glance

| | |
|---|---|
| **Domain** | Healthcare front-desk automation (specialty medical practices) |
| **AI surfaces** | LLM chat &middot; Voice (STT + TTS) &middot; Vision OCR &middot; Multi-agent orchestration |
| **Architecture** | LangGraph multi-agent backend, FastAPI service, Gradio dashboard, SQLite + FAISS |
| **Tenants shipped** | 2 fully-configured (ophthalmology, orthopedics) &mdash; same code, different YAML |
| **LLM stack** | gpt-4o-mini (chat + vision) &middot; whisper-1 (STT) &middot; tts-1 (TTS) &middot; text-embedding-3-small (RAG) |
| **Code scale** | ~20,000 lines across `clarion/`, `gradio_app/`, `api/`, `tests/` |
| **Quality gates** | 705 tests, mypy `--strict` on agent core, ruff clean, locked schema regression test |
| **Production primitives** | Retry-with-jitter, circuit breaker, token-bucket rate limit, correlation IDs, structured JSON logs |
| **Evaluation** | 100-scenario synthetic corpus per tenant, locked schema v1.0.0, regression-gated at CI |

---

## Screenshots

| | |
|:--:|:--:|
| ![Sentinel Operations](docs/screenshots/03-sentinel-ops.png) <br/> **Sentinel Operations** &mdash; Independent trust engine grading every agent reply. Composite trust gauge + five-signal escalation breakdown. | ![Agent Flow](docs/screenshots/04-agent-flow.png) <br/> **Agent Flow** &mdash; Live trace through the multi-agent LangGraph. Router &rarr; specialist &rarr; supervisor with the executed path highlighted. |
| ![Voice Intelligence](docs/screenshots/05-voice-intelligence.png) <br/> **Voice Intelligence** &mdash; Emotion distribution, frustration trace, and per-turn escalation prediction lifted from voice transcripts. | ![Patient 360](docs/screenshots/06-patient-360.png) <br/> **Patient 360** &mdash; Unified longitudinal record with engagement, sentiment, and trust scores; care team and insurance below. |
| ![Cost & SLO](docs/screenshots/07-cost-slo.png) <br/> **Cost & SLO** &mdash; Per-tenant spend, latency budgets, cost-share donut. The operational control plane. | ![Invoice OCR](docs/screenshots/07-cost-slo-ocr.png) <br/> **Invoice OCR** &mdash; gpt-4o-mini Vision pipeline that lifts every line item from a vendor invoice into a structured row with running total. |
| ![Live Agent](docs/screenshots/08-live-agent.png) <br/> **Live Agent** &mdash; Direct interface to the production LangGraph agent. Tool calls, escalation score, and cost surfaced per turn. | ![Voice Agent](docs/screenshots/09-voice-agent.png) <br/> **Voice Agent** &mdash; End-to-end voice round-trip. Whisper STT &rarr; same agent core &rarr; OpenAI TTS, with per-stage latency surfaced. |

---

## The Problem

Specialty medical practices lose patients and revenue at the first touchpoint &mdash; the phone. Front-desk teams are overwhelmed by high-volume routine traffic (bookings, eligibility checks, payer questions) while needing zero tolerance for failure on critical calls: sudden vision loss, suspected fracture, or a patient soliciting clinical advice.

Voice AI is the obvious load-shedding solution, but a generic LLM agent fails three tests that matter:

1. **Auditability.** When the agent gets something wrong, the operator must reconstruct exactly what happened, with evidence, in seconds. Most agent frameworks produce opaque traces.
2. **Safety bounds.** The system must refuse clinical advice and escalate emergencies *before* the LLM replies. Pure-LLM moderation runs after the model has already seen the message.
3. **Multi-tenancy without code branches.** One client allows appointment cancellation through the bot; another routes cancels to a human. Solving this with `if customer_id == "..."` fragments quickly and becomes unreviewable.

Clarion makes all three structurally impossible to get wrong.

---

## System Architecture

```
                 ┌─────────────────────────────────────────┐
                 │       11-tab Gradio dashboard            │
                 │  reads only clarion.schemas (typed)      │
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

**Five layers, each with a one-way dependency on the layer below.** Layer boundaries are enforced by the Python module graph — the dashboard cannot import the evaluation pipeline, so it is structurally impossible for the UI to recompute a metric. A new metric is always, and only, a backend change.

| Layer | Responsibility |
|---|---|
| **Dashboard** | Gradio Blocks app. 11 tabs of pure HTML rendered from typed dataclasses. Read-only consumer of the schema package. |
| **API + harness** | FastAPI service (`/chat`, `/voice/turn`, `/cost/extract-invoice`) and the 100-scenario evaluation harness that produces the locked report. |
| **Sentinel** | Independent trust engine. Pre-LLM guardrails, post-LLM judge, escalation scorer. Failure-mode aware. |
| **Multi-agent core** | LangGraph router &rarr; 5 specialists &rarr; supervisor. Per-specialist tool scoping shrinks the attack surface. |
| **Foundation** | Per-tenant YAML config, SQLite structured store, FAISS-backed RAG, typed tool registry. |

Full architecture diagram: [docs/architecture.mmd](docs/architecture.mmd) (Mermaid, text-diffable) / [docs/architecture.png](docs/architecture.png) (rendered PNG).

---

## Evaluation & Results

Clarion ships a **100-scenario synthetic evaluation corpus** per tenant, driven by a scripted `FakeLLM` (deterministic, no API key required in CI). The report schema is locked at `v1.0.0` and regression-gated at every CI run.

### Evaluation Metrics — Both Tenants

| Metric | Ophthalmology | Orthopedics |
|:--|:--:|:--:|
| **Scenarios evaluated** | 100 | 100 |
| **Pass rate** | **100.0%** | **100.0%** |
| **Booking accuracy** | **100.0%** (38 / 38) | **100.0%** (38 / 38) |
| **Safety catch rate** (emergencies) | **100.0%** (20 / 20) | **100.0%** (20 / 20) |
| **Escalation recall** | **100.0%** | **100.0%** |
| **Escalation precision** | 61.5% | 78.7% |
| **Escalation F1** | 76.2% | 88.1% |
| **Escalation accuracy** | 90.0% | 90.0% |
| **Containment rate** | 74.0% | 66.0% |
| **Hallucinations (LLM-as-Judge)** | 0 | 0 |
| **Eval schema version** | v1.0.0 | v1.0.0 |
| **Generated at** | 2026-06-23 | 2026-06-23 |

> **Metric definitions:**
> - *Pass rate* — did the agent complete the scenario correctly end-to-end?
> - *Booking accuracy* — of the 38 booking scenarios per tenant, how many booked the correct slot with all fields valid?
> - *Safety catch rate* — of the 20 emergency/safety-critical scenarios, how many were escalated *before* any clinical advice was given? 20/20 = the LLM was never invoked on a safety-critical turn.
> - *Escalation recall* — did the system escalate every scenario it was supposed to? 100% = zero missed escalations.
> - *Containment rate* — fraction of routine calls the agent resolved without human escalation.
> - *Hallucinations* — cases where the LLM-as-Judge (Sentinel) detected a factually incorrect or out-of-policy response.

### System-Level Results

| Metric | Value | Source |
|---|---|---|
| Tests passing | **705 / 705** | `poetry run pytest` |
| Test suite runtime | ~50 s | full pytest run |
| Mypy strict | clean on `clarion`, `api` | CI gate |
| Ruff | clean | CI gate |
| Code volume | ~20,000 LOC | `clarion/`, `gradio_app/`, `api/`, `tests/` |
| Container cold start | < 2 s | multi-stage Docker, pre-baked FAISS |
| In-process p50 latency | < 200 ms | `tests/loadtest/test_p95_sla.py` (FakeLLM) |
| In-process p95 latency | < 500 ms | `tests/loadtest/test_p95_sla.py` (FakeLLM) |
| BookingFastPath held-out accuracy | **≥ 90%** | `BookingFastPath.train_default()` |

### Running the Evaluation

```bash
# Scripted mode — no API key, deterministic, used in CI
poetry run python -m clarion.eval --customer all

# Live mode — real gpt-4o-mini calls, ~$1 for both tenants combined
poetry run python -m clarion.eval --customer all --mode live

# Single tenant
poetry run python -m clarion.eval --customer ophthalmology
```

The locked report schema (`EvaluationReport` v1.0.0) is enforced by
[`tests/schemas/test_evaluation_schema_lock.py`](tests/schemas/test_evaluation_schema_lock.py) — additive changes pass, breaking changes fail CI.

---

## Engineering Decisions

The decisions below are load-bearing — the parts of the design the rest of the system relies on remaining true.

### 1. Schema-first contracts at every boundary

Every cross-module communication is a Pydantic v2 model with `extra="forbid"`. The agent's tool inputs, the FastAPI request/response bodies, the locked evaluation report, the wire models the dashboard reads — all typed, all validated at the boundary, all impossible to bypass without an `import` change a reviewer would notice.

### 2. Three-layer boundary input validation

Tool inputs go through three independent validation layers:

```python
# clarion/schemas/tools.py
class BookAppointmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_id       : str = Field(min_length=1, pattern=_ID_PATTERN)
    patient_id    : str = Field(min_length=1, pattern=_ID_PATTERN)
    patient_name  : str = Field(pattern=_NAME_PATTERN)   # 2+ words; intl chars OK
    patient_phone : str = Field(pattern=_PHONE_PATTERN)  # 7-25 chars, digit-rich
    patient_email : str = Field(pattern=_EMAIL_PATTERN)  # local@domain.tld
```

Layer 1 is the booking specialist's system prompt (reads each field back to the caller before invoking). Layer 2 is the regex above. Layer 3 is the structured store, which persists the values as JSON so Patient 360 renders exactly what the caller confirmed.

The regex layer was added after the agent hallucinated a caller's full name into the `patient_id` field, which the store accepted silently. Tests in [`tests/tools/test_book_appointment.py`](tests/tools/test_book_appointment.py) lock in rejection of that input class. See commit [`a36a72f`](https://github.com/Ranjith200228/clarion/commit/a36a72f).

### 3. Config-driven multi-tenancy and per-specialist tool scoping

```yaml
# configs/orthopedics.yaml
enabled_tools:
  - search_slots
  - book_appointment
  - check_eligibility
  - create_pms_task
  # cancel_appointment intentionally omitted
```

A specialist's LLM never sees a tool that wasn't advertised to it. The Info specialist literally cannot call `book_appointment`. The Emergency specialist is deterministic — no LLM call, no tool advertisement, no risk of clever talking. Prompt-injection attempts have a strictly smaller attack surface than in a monolithic ReAct loop.

### 4. Independent trust engine (Sentinel)

Sentinel grades every agent reply from outside the agent's perspective:

| Component | Output | Failure mode |
|---|---|---|
| Guardrails (emergency / clinical / PHI) | Short-circuits the reply — LLM never invoked | Pattern-based; biases toward false alarms over silent failures |
| LLM-as-Judge (booking + hallucination + policy) | Structured verdict per turn | Defensive JSON parsing; defaults to low-confidence on malformed output |
| Escalation scorer | 0&ndash;1 score fusing 5 weighted signals | Tunable thresholds in tenant YAML for per-tenant calibration |

Because the judge runs *after* the agent, it does not share the agent's blind spots. Two real bugs surfaced during evaluation: a hallucinated specialty on a tenant that doesn't offer it, and a booking that confirmed a time inconsistent with the reserved slot. Both would have looked correct to a human reviewer; both were caught because the judge had no skin in the game.

### 5. Locked evaluation contract (v1.0.0)

`EvaluationReport` is the wire model the dashboard reads. Schema version is locked at `1.0.0`. New fields must be optional. Existing field names and types cannot change. A regression test gates this at CI.

Result: the dashboard never re-implements a metric. Every metric definition exists in exactly one place (`clarion.evaluation.metrics`) and propagates through the contract to every consumer. When a definition changes, every view updates simultaneously.

### 6. Multi-modal AI through Protocol adapters

STT, TTS, and the LLM client each sit behind a one-method Protocol:

```python
class TranscriberProtocol(Protocol):
    def transcribe(self, audio_bytes: bytes, metadata: AudioMetadata) -> str: ...
```

Production uses `FasterWhisperTranscriber` and `OpenAITtsSpeaker`; tests use `EchoTranscriber` and `SineWaveSpeaker`. The agent implementation is unaware of which is mounted — CI never needs an API key and the deployment can swap providers without touching the agent loop.

The invoice OCR module ([`clarion/modules/invoice_ocr.py`](clarion/modules/invoice_ocr.py)) follows the same pattern: gpt-4o-mini Vision called in JSON-mode with a strict extraction prompt; defensive response parsing strips code fences and coerces number-strings; the result is the same typed dataclass shape the rest of the app uses. Adding the entire multimodal capability cost one Python module, one FastAPI route, one Gradio component, one CSS block.

### 7. Production reliability primitives

| Concern | Implementation | Config |
|---|---|---|
| Transient API failures | Retry with full-jitter exponential backoff | 4 attempts, base 0.25 s, cap 8 s |
| Cascading upstream failures | Circuit breaker around the LLM client | 5 failures → 30 s open state |
| Tenant noisy-neighbor | Per-`(tenant, IP)` token-bucket rate limit | 10 rps, burst 30 |
| Request observability | Correlation ID middleware + structured JSON logs | `X-Request-Id` header echo |
| Cold-start latency | Multi-stage Docker, FAISS pre-baked in builder stage | Container ready in < 2 s |
| Health & readiness | `/health` endpoint, `HEALTHCHECK` directive | Liveness on the container |

### 8. BookingFastPath: a 3 ms intent classifier

The `BookingSpecialist` runs a TF-IDF + Logistic Regression classifier on every user turn *before* invoking the LLM. The classifier maps the message to one of five booking intents (search / book / reschedule / cancel / check\_eligibility) or a fallback bucket. When confidence clears 0.65, a one-line hint is prepended to the specialist's system prompt — guidance the LLM uses to skip the intent-discovery phase entirely.

The model trains in < 1 s from 480 synthetic utterances, runs in under 3 ms per turn, and achieves ≥ 90% accuracy on a held-out split. That hint drops 40–80 prompt tokens and one reasoning step per turn — tangible token savings at conversation volume.

---

## Dashboard — 11 Tabs

| Tab | What it shows |
|---|---|
| **Mission Control** | System health overview: total bookings, escalations, safety events, and average trust score across all tenants. |
| **Patient 360** | Unified longitudinal record: engagement timeline, sentiment trend, trust score, care team, and insurance details. |
| **Sentinel Operations** | Independent trust engine view: composite trust gauge, five-signal escalation breakdown, per-turn judge verdicts. |
| **Agent Flow** | Live LangGraph trace: which specialist ran, per-node timing, which tools fired, full supervisor routing. |
| **Voice Intelligence** | Emotion distribution, frustration trace, and per-turn escalation prediction lifted from voice transcripts. |
| **Healthcare Operations** | Domain metrics: no-show prediction, booking funnel, payer mix, appointment type distribution. |
| **Cost & SLO** | Per-tenant spend, latency budgets, token-count trends, cost-share donut. |
| **Evaluation Summary** | Full locked-schema eval report: pass rate, booking accuracy, safety catch rate, escalation metrics, by-difficulty breakdown. |
| **Live Agent** | Direct interface to the production LangGraph agent. Tool calls, escalation score, and cost per turn. |
| **Voice Agent** | End-to-end voice round-trip: Whisper STT → agent core → OpenAI TTS, per-stage latency surfaced. |
| **Invoice OCR** | gpt-4o-mini Vision pipeline: upload a vendor invoice PDF or image, get every line item as a structured table. |

---

## Customer Onboarding

A new tenant is a single YAML file. The six-step recipe:

1. **Copy** `configs/ophthalmology.yaml` → `configs/new_practice.yaml`
2. **Edit** `name`, `enabled_tools`, `rules_path`, escalation thresholds, and persona string.
3. **Add** practice-specific rules as Markdown files under `data/new_practice/rules/`.
4. **Register** the YAML path in `configs/__init__.py` (two lines).
5. **Run** `poetry run python -m clarion.eval --customer new_practice` to populate evaluation artifacts.
6. **Verify** the locked report and spot-check the Gradio customer switcher.

No agent code changes. No new tool registrations. No new tests beyond running the existing harness. Full recipe in [docs/developer_guide.md](docs/developer_guide.md).

---

## Quick Start

### Local Development

```bash
git clone https://github.com/Ranjith200228/clarion.git
cd clarion
poetry install
export OPENAI_API_KEY=sk-...

# Generate evaluation artifacts (scripted mode — no API key needed)
poetry run python -m clarion.eval --customer all

# Terminal 1 — FastAPI backend
poetry run uvicorn api.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — Gradio dashboard
poetry run python -m gradio_app

# Open http://localhost:7860
```

The dashboard works without the API for every tab that reads artifacts off disk. Only **Live Agent**, **Voice Agent**, and **Invoice OCR** require the backend running.

### Docker

```bash
docker compose up    # API + dashboard in a single container, production parity
```

The multi-stage Dockerfile pre-bakes FAISS indices in the builder stage; a fresh container serves requests in under 2 seconds.

### Run Quality Gates

```bash
poetry run pytest                              # all 705 tests
poetry run ruff check clarion gradio_app api tests
poetry run mypy --strict clarion api
poetry run pytest -m loadtest                  # opt-in p95 SLA budget
```

---

## Tech Stack

| Layer | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Type-system maturity, async/await ergonomics, ecosystem coverage |
| **Validation** | Pydantic v2 | Performance, JSON-schema generation for OpenAI tool calls, `extra="forbid"` discipline |
| **LLM (chat + vision)** | OpenAI `gpt-4o-mini` | Best price-to-capability for the task; native tool calling; JSON mode for OCR |
| **STT / TTS** | `whisper-1` / `tts-1` | Production-quality on the same vendor stack; minimal integration surface |
| **Multi-agent** | LangGraph (StateGraph) | First-class explicit-graph semantics; auditable supervisor logic |
| **API** | FastAPI | ASGI performance, automatic OpenAPI from Pydantic models |
| **UI** | Gradio 4.44 Blocks + custom CSS | Fastest path from typed dataclass to interactive surface; ~1,300-line design-token CSS layer |
| **Storage** | SQLite per-tenant + FAISS | Zero-ops local store, deterministic for evaluation; FAISS for semantic search |
| **ML** | XGBoost no-show classifier + TF-IDF/LogReg FastPath | Held-out ROC-AUC and top-decile lift in the locked report; 3 ms intent classifier |
| **Trust engine** | Hand-rolled | Off-the-shelf moderation libraries hide their failure modes; this one's are explicit |
| **Deploy** | Multi-stage Docker, Hugging Face Spaces | Reproducible builds; manifests also ship for Cloud Run, Render, Fly.io |
| **CI** | GitHub Actions (3.11 + 3.12 matrix) | pytest, ruff, mypy strict, schema regression test |

---

## Repository Structure

```
clarion/
  agents/             Single-agent ReAct loop + OpenAI client wrapper
  multiagent/         LangGraph backend: router, specialists, supervisor
    booking_fastpath.py   TF-IDF + LogReg intent classifier (3 ms / turn)
  sentinel/           Trust engine — guardrails, judge, escalation scorer, PHI
  schemas/            Pydantic wire models — the contract layer between modules
  modules/
    invoice_ocr.py        gpt-4o-mini Vision invoice extraction
    no_show_prediction/   XGBoost classifier persisted to joblib
    pms_writeback/        Conversation → summary.json + task.json (PHI-redacted)
    voice/                Whisper STT + OpenAI TTS round trip
  pipelines/          Structured store (SQLite) + RAG retriever (FAISS)
  resilience/         Retry-with-jitter, circuit breaker, rate limit
  evaluation/         100-scenario harness, locked-schema report writer
  observability/      Structured JSON logs, correlation IDs, spans
  config/             Settings + per-tenant YAML loader

api/
  app.py              FastAPI factory
  routes/             /chat  /voice/turn  /cost/extract-invoice  /health
  middleware/         Correlation IDs + token-bucket rate limiter
  sessions.py         Per-(tenant, conversation) session manager

gradio_app/
  app.py              11-tab Blocks shell + customer switcher
  views/              One file per tab — pure HTML render functions
  components.py       Shared visual primitives (KPI tile, donut, page intro)
  data_sources.py     Typed roll-ups consumed by the views
  tab_*.py            Stateful tabs (live agent, voice agent, invoice OCR)
  theme.py / style.css   Design tokens and CSS primitives (~1,300 lines)

configs/              Per-tenant YAML (single source of customer behavior)
data/                 Per-tenant artifacts (eval reports, trace files, rules)
tests/                705 pytest tests (unit + integration + e2e)
docs/                 Discovery doc, dev guide, deploy guide, security review
loadtest/             Locust profile + in-process p95 SLA test
scripts/              Build helpers, index builder, gradio patch, healthcheck
```

---

## Deployment

```bash
# Local dev
poetry install && poetry run python -m clarion.eval --customer all
poetry run uvicorn api.app:app --port 8000 &
poetry run python -m gradio_app

# Docker (single container — HF Spaces / Cloud Run / Render / Fly.io)
docker build -t clarion .
docker run -e OPENAI_API_KEY=sk-... -p 7860:7860 clarion
```

**Live deployment:** [huggingface.co/spaces/Ranjithmaddirala/clarion](https://huggingface.co/spaces/Ranjithmaddirala/clarion)

Full deployment guide (Cloud Run, Render, Fly.io, Hugging Face Spaces): [docs/deployment_guide.md](docs/deployment_guide.md).

---

## Lessons Learned

**Prompt engineering alone doesn't scale.** The original single-agent loop required a 600-token system prompt to constrain tool selection, intent routing, and safety rules simultaneously. Splitting into specialist nodes lets each node carry a tight, purpose-built prompt; the BookingFastPath classifier is the next step — shrinking even that further by pre-routing with a 3 ms classifier instead of 40–80 tokens of intent-discovery guidance.

**Schema locks make refactors safe.** Locking `EvaluationReport` at v1.0.0 turned out to be the most operationally valuable decision in the project. Three separate metric refactors would have silently broken the dashboard if the schema gate hadn't caught them first.

**The trust engine finds bugs the agent doesn't.** The LLM-as-Judge running outside the agent's context caught two production-level bugs during evaluation — a hallucinated specialty and a time inconsistency — that both looked correct from inside the agent loop. An independent grader is worth the latency cost.

**Independent calibration matters per tenant.** Escalation thresholds that worked for ophthalmology (where "my eye hurts" is clinically urgent) produced false alarms for orthopedics. Tenant-level YAML config for thresholds, rather than hardcoded values, removed an entire class of per-tenant hotfixes.

---

## Reading Deeper

| Document | What's in it |
|---|---|
| [docs/discovery.md](docs/discovery.md) | Customer-problem framing, requirements, and the architecture's mapping onto them |
| [docs/developer_guide.md](docs/developer_guide.md) | Local setup, internal layout, and the six-step new-tenant onboarding flow |
| [docs/deployment_guide.md](docs/deployment_guide.md) | Container build, secret management, and the four supported hosting targets |
| [docs/security_review.md](docs/security_review.md) | STRIDE-shaped audit including the gaps list to HIPAA |
| [CHANGELOG.md](CHANGELOG.md) | What shipped, in order, with code pointers |

---

## Roadmap

| Milestone | Status |
|---|---|
| v1.0.0 release with locked evaluation schema | [shipped 2026-06-12](https://github.com/Ranjith200228/clarion/releases/tag/v1.0.0) |
| LangGraph multi-agent backend | shipped |
| PMS writeback module with PHI redaction | shipped |
| XGBoost no-show prediction folded into the report | shipped |
| Voice layer (Whisper + OpenAI TTS) | shipped |
| Production hardening (retry, breaker, rate limit, structured logs) | shipped |
| Invoice OCR via gpt-4o-mini Vision | shipped |
| Three-layer patient detail validation | shipped |
| BookingFastPath classifier — 3 ms intent priming, ≥ 90% accuracy | shipped |
| **Next**: custom domain in front of the Hugging Face Space | planned |
| **Next**: online learning — retire synthetic training data for real call transcripts | planned |

---

[MIT](LICENSE) &copy; 2026 Ranjith Maddirala

I'm always interested in discussing the design decisions in this project — particularly the trust engine, the schema-locked contract, and the multi-agent tool-scoping. Reach me at **ranjithmaddirala24@gmail.com**.
