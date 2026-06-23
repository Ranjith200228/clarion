<div align="center">

# Clarion

**A production-grade, multi-tenant, multi-agent voice + vision platform for healthcare operations.**

Engineered around three load-bearing principles: typed contracts at
every boundary, an independent trust engine that grades the AI from
outside, and config-driven multi-tenancy so onboarding a new vertical
never touches the agent loop.

[![Tests](https://img.shields.io/badge/tests-580%20passing-22D3EE?style=flat-square)](#testing--quality-gates)
[![Python](https://img.shields.io/badge/python-3.11%2B-22D3EE?style=flat-square)](pyproject.toml)
[![mypy](https://img.shields.io/badge/mypy-strict-22D3EE?style=flat-square)](#testing--quality-gates)
[![Ruff](https://img.shields.io/badge/ruff-clean-22D3EE?style=flat-square)](#testing--quality-gates)
[![Coverage](https://img.shields.io/badge/coverage-19.6k%20LOC-22D3EE?style=flat-square)](#repository-structure)
[![License](https://img.shields.io/badge/license-MIT-22D3EE?style=flat-square)](LICENSE)
[![Demo](https://img.shields.io/badge/demo-Hugging%20Face%20Space-FFD21E?style=flat-square)](huggingface/README.md)

**[Live demo](huggingface/README.md)**
&nbsp;&middot;&nbsp; **[Architecture](#system-architecture)**
&nbsp;&middot;&nbsp; **[Engineering decisions](#engineering-decisions)**
&nbsp;&middot;&nbsp; **[Performance](#performance--reliability)**
&nbsp;&middot;&nbsp; **[Run it](#quick-start)**

</div>

---

## At a glance

| | |
|---|---|
| **Domain** | Healthcare front-desk automation (specialty medical practices) |
| **AI surfaces** | LLM chat &middot; Voice (STT + TTS) &middot; Vision OCR &middot; Multi-agent orchestration |
| **Architecture** | LangGraph multi-agent backend, FastAPI service, Gradio dashboard, SQLite + FAISS storage |
| **Tenants shipped** | 2 fully-configured (ophthalmology, orthopedics) &mdash; same code, different YAML |
| **LLM stack** | gpt-4o-mini (chat + vision) &middot; whisper-1 (STT) &middot; tts-1 (TTS) &middot; text-embedding-3-small (RAG) |
| **Scale of code** | ~19,600 lines across `clarion/`, `gradio_app/`, `api/`, `tests/` |
| **Quality gates** | 580 tests, mypy `--strict` on the agent core, ruff clean, CI matrix on 3.11 + 3.12 |
| **Production primitives** | Retry-with-jitter, circuit breaker, token-bucket rate limit, correlation IDs, structured JSON logs |
| **Evaluation** | 100-scenario synthetic corpus per tenant, locked schema v1.0.0, regression-tested |

---

## Screenshots

| | |
|:--:|:--:|
| ![Sentinel Operations](docs/screenshots/03-sentinel-ops.png) <br/> **Sentinel Operations** &mdash; Independent trust engine grading every agent reply. Composite trust gauge + five-signal escalation breakdown. | ![Agent Flow](docs/screenshots/04-agent-flow.png) <br/> **Agent Flow** &mdash; Live trace through the multi-agent LangGraph. Router &rarr; specialist &rarr; supervisor with the executed path highlighted. |
| ![Voice Intelligence](docs/screenshots/05-voice-intelligence.png) <br/> **Voice Intelligence** &mdash; Emotion distribution, frustration trace, and per-turn escalation prediction lifted from voice transcripts. | ![Patient 360](docs/screenshots/06-patient-360.png) <br/> **Patient 360** &mdash; Unified longitudinal record with engagement, sentiment, and trust scores; care team and insurance below. |
| ![Cost & SLO](docs/screenshots/07-cost-slo.png) <br/> **Cost & SLO** &mdash; Per-tenant spend, latency budgets, cost share donut. The operational control plane. | ![Invoice OCR](docs/screenshots/07-cost-slo-ocr.png) <br/> **Invoice OCR** &mdash; gpt-4o-mini Vision pipeline that lifts every line item from a vendor invoice into a structured row with running total. |
| ![Live Agent](docs/screenshots/08-live-agent.png) <br/> **Live Agent** &mdash; Direct interface to the production LangGraph agent. Tool calls, escalation score, and cost surface per turn. | ![Voice Agent](docs/screenshots/09-voice-agent.png) <br/> **Voice Agent** &mdash; End-to-end voice round-trip. Whisper STT &rarr; same agent core &rarr; OpenAI TTS, with per-stage latency surfaced. |

---

## The problem

Specialty medical practices lose patients and revenue at the first
touchpoint &mdash; the phone. Front-desk teams are overwhelmed by
high-volume routine traffic (bookings, eligibility checks, payer
questions) while needing zero tolerance for failure on critical
calls: a sudden vision loss, suspected fracture, or a patient
soliciting clinical advice.

Voice AI is the obvious load-shedding solution, but a generic LLM
agent fails three tests that matter:

1. **Auditability.** When the agent gets something wrong, the
   operator must be able to reconstruct exactly what happened, with
   evidence, in seconds. Most agent frameworks produce opaque traces.
2. **Safety bounds.** The system must refuse to give clinical advice
   and must escalate emergencies before the LLM even has a chance to
   reply. Pure-LLM moderation runs after the model has already seen
   the message.
3. **Multi-tenancy without code branches.** One client allows
   appointment cancellation through the bot; another routes cancels
   to a human task. Solving this with `if customer_id == "..."`
   fragments quickly and is unreviewable.

Clarion was designed to make all three structurally impossible to
get wrong.

---

## System architecture

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

**Five layers, each with a one-way dependency on the layer below.**
Layer boundaries are enforced by the Python module graph &mdash; the
dashboard cannot import the evaluation pipeline, so it is
structurally impossible for the UI to recompute a metric. A new
metric is always, and only, a backend change.

| Layer | Responsibility |
|---|---|
| **Dashboard** | Gradio Blocks app. 11 tabs of pure HTML rendered from typed dataclasses. Read-only consumer of the schema package. |
| **API + harness** | FastAPI service (`/chat`, `/voice/turn`, `/cost/extract-invoice`) and the 100-scenario evaluation harness that produces the locked report. |
| **Sentinel** | Independent trust engine. Pre-LLM guardrails, post-LLM judge, escalation scorer. Failure-mode aware. |
| **Multi-agent core** | LangGraph router &rarr; 5 specialists &rarr; supervisor. Per-specialist tool scoping shrinks the attack surface. |
| **Foundation** | Per-tenant YAML config, SQLite structured store, FAISS-backed RAG, typed tool registry. |

---

## Engineering decisions

The decisions below are the load-bearing ones &mdash; the parts of
the design that the rest of the system relies on remaining true.

### 1. Schema-first contracts at every boundary

Every cross-module communication is a Pydantic v2 model with
`extra="forbid"`. The agent's tool inputs, the FastAPI request /
response bodies, the locked evaluation report, the wire models the
dashboard reads &mdash; all typed, all validated at the boundary,
all impossible to bypass without an `import` change a reviewer would
notice.

### 2. Boundary input validation: defense in depth

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

Layer 1 is the booking-specialist's system prompt, which requires the
agent to read each field back to the caller before invoking the tool.
Layer 2 is the regex above. Layer 3 is the structured store, which
persists the values into the appointment's `notes` column as JSON
so the Patient 360 confirmation card renders what the caller
actually confirmed.

The regex layer was added after the agent hallucinated a caller's
full name into the `patient_id` field, which the store accepted and
the dashboard then rendered. Tests in
[`tests/tools/test_book_appointment.py`](tests/tools/test_book_appointment.py)
lock in the rejection of the original input. The relevant commit:
[`a36a72f`](https://github.com/Ranjith200228/clarion/commit/a36a72f).

### 3. Per-specialist tool advertisement

The multi-agent backend isolates blast radius by specialist:

```yaml
# configs/orthopedics.yaml
enabled_tools:
  - search_slots
  - book_appointment
  - check_eligibility
  - create_pms_task
  # cancel_appointment intentionally omitted
```

A specialist's LLM never sees a tool that wasn't advertised to it.
The Info specialist literally cannot call `book_appointment` because
that tool was never on its tool list. The Emergency specialist is
deterministic &mdash; no LLM call, no tool advertisement, no risk
of clever talking. Prompt-injection attempts have a strictly smaller
attack surface than in a monolithic ReAct loop.

### 4. Independent trust engine

Sentinel grades every agent reply from outside the agent's
perspective. Three components, each with an explicit failure mode:

| Component | Output | Failure mode |
|---|---|---|
| Guardrails (emergency / clinical / PHI) | Short-circuits the reply &mdash; LLM never invoked | Pattern-based; biases toward false alarms over silent failures |
| LLM-as-Judge (booking + hallucination + policy) | Structured verdict per turn | Defensive JSON parsing; defaults to low-confidence on malformed output |
| Escalation scorer | 0&ndash;1 score fusing 5 weighted signals | Tunable thresholds in tenant YAML for per-tenant calibration |

Because the judge runs *after* the agent, it does not share the
agent's blind spots. Two real bugs surfaced during evaluation: a
hallucinated specialty on a tenant that doesn't offer it, and a
booking that confirmed a time inconsistent with the reserved slot.
Both would have looked correct to a human reviewer; both were caught
because the judge had no skin in the game.

### 5. Locked evaluation contract (v1.0.0)

`EvaluationReport` is the wire model the dashboard reads. Schema
version is locked at `1.0.0`. New fields must be optional. Existing
field names and types cannot change. A regression test
([`tests/schemas/test_evaluation_schema_lock.py`](tests/schemas/test_evaluation_schema_lock.py))
gates this at CI.

Result: the dashboard never re-implements a metric. Every metric
definition exists in exactly one place (`clarion.evaluation.metrics`)
and propagates through the contract to every consumer. When a
definition changes, every view updates simultaneously.

### 6. Multi-modal AI integration through Protocol adapters

STT, TTS, and the LLM client each sit behind a one-method Protocol:

```python
class TranscriberProtocol(Protocol):
    def transcribe(self, audio_bytes: bytes, metadata: AudioMetadata) -> str: ...
```

Production uses `FasterWhisperTranscriber` and `OpenAITtsSpeaker`;
tests use `EchoTranscriber` and `SineWaveSpeaker`. The agent
implementation is unaware of which is mounted, which means CI never
needs an API key and the deployment can swap providers without
touching the agent loop.

The invoice OCR module
([`clarion/modules/invoice_ocr.py`](clarion/modules/invoice_ocr.py))
follows the same pattern: gpt-4o-mini Vision called in JSON-mode with
a strict extraction prompt; defensive response parsing strips code
fences and coerces number-strings; the result is the same typed
dataclass shape the rest of the app uses. Adding the entire
multimodal capability cost one Python module, one FastAPI route, one
Gradio component, one CSS block.

### 7. Production-grade reliability primitives

| Concern | Implementation | Configuration |
|---|---|---|
| Transient API failures | Retry with full-jitter exponential backoff | 4 attempts, base 0.25s, cap 8s |
| Cascading upstream failures | Circuit breaker around the LLM client | 5 failures &rarr; 30s open state |
| Tenant noisy-neighbor | Per-`(tenant, IP)` token-bucket rate limit | 10 rps, burst 30 |
| Request observability | Correlation ID middleware + structured JSON logs | `X-Request-Id` header echo |
| Cold-start latency | Multi-stage Docker, FAISS pre-baked in builder stage | Container ready in <2s |
| Health & readiness | `/health` endpoint, `HEALTHCHECK` directive | Liveness on the container |

---

## Performance & reliability

| Metric | Value | Source |
|---|---|---|
| Tests passing | **580 / 580** | `poetry run pytest` |
| Test suite runtime | ~30s | full pytest run |
| Mypy strict | clean on `clarion`, `api` | CI gate |
| Ruff | clean | CI gate |
| Code volume | ~19,600 LOC | `clarion/`, `gradio_app/`, `api/`, `tests/` |
| Container cold start | < 2s | Multi-stage Docker, pre-baked FAISS |
| In-process p50 | < 200 ms | `tests/loadtest/test_p95_sla.py` with `FakeLLM` |
| In-process p95 | < 500 ms | `tests/loadtest/test_p95_sla.py` with `FakeLLM` |
| Booking accuracy (ophthalmology) | **100%** (38/38) | locked report v1.0.0 |
| Booking accuracy (orthopedics) | **100%** (38/38) | locked report v1.0.0 |
| Safety catch rate (emergencies) | **100%** | locked report v1.0.0 |
| Escalation recall | **100%** | locked report v1.0.0 |

---

## Tech stack

| Layer | Choice | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Type-system maturity, async/await ergonomics, ecosystem coverage. |
| **Validation** | Pydantic v2 | Performance, JSON-schema generation for OpenAI tool calls, strict `extra="forbid"` discipline. |
| **LLM (chat + vision)** | OpenAI `gpt-4o-mini` | Best price-to-capability for the task; native tool calling; JSON mode for OCR. |
| **STT / TTS** | `whisper-1` (STT), `tts-1` (TTS) | Production-quality on the same vendor stack; minimal integration surface. |
| **Multi-agent** | LangGraph (StateGraph) | First-class explicit-graph semantics; auditable supervisor logic. |
| **API** | FastAPI | ASGI performance, automatic OpenAPI from Pydantic models, ecosystem. |
| **UI** | Gradio 4.44 Blocks + custom CSS | Fastest path from typed dataclass to interactive surface; design-token CSS layer (~1,300 lines) for visual polish. |
| **Storage** | SQLite per-tenant + FAISS | Zero-ops local store, deterministic for evaluation; FAISS for semantic search at scale. |
| **ML** | XGBoost no-show classifier | Held-out ROC-AUC and top-decile lift folded into the locked report; synthetic-data only. |
| **Trust engine** | Hand-rolled | Off-the-shelf moderation libraries hide their failure modes; this one's are explicit. |
| **Deploy** | Multi-stage Docker, Hugging Face Spaces (primary) | Reproducible builds; manifests also ship for Cloud Run, Render, Fly.io. |
| **CI** | GitHub Actions (3.11 + 3.12 matrix) | Pytest, ruff, mypy strict, schema regression test. |

---

## Quick start

```bash
git clone https://github.com/Ranjith200228/clarion.git
cd clarion
poetry install
export OPENAI_API_KEY=sk-...

# Populate evaluation artifacts (synthetic; run once)
poetry run python -m clarion.eval --customer all

# Terminal 1 — FastAPI backend
poetry run python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# Terminal 2 — Gradio dashboard
poetry run python -m gradio_app

# Open http://localhost:7860
```

The dashboard works without the API for everything that reads
artifacts off disk &mdash; only **Live Agent**, **Voice Agent**, and
**Invoice OCR** need the backend running.

```bash
docker compose up    # API + dashboard side-by-side, production parity
```

The multi-stage Dockerfile pre-bakes FAISS indices in the builder
stage; a fresh container serves requests in under two seconds.

---

## Testing & quality gates

```bash
poetry run pytest                              # the full 580
poetry run ruff check clarion gradio_app api tests
poetry run mypy --strict clarion api
poetry run pytest -m loadtest                  # opt-in p95 SLA budget
```

Coverage highlights:

- **End-to-end booking flows** for both tenants, driven by a scripted
  `FakeLLM` so CI requires no API key.
- **Boundary regex guards** on `BookAppointmentInput` &mdash; six
  cases covering single-word names, "ask me later" phones, "n/a"
  emails, plus acceptance of international and E.164 formats.
- **Schema regression** on the locked `EvaluationReport` v1.0.0
  contract. Additive changes pass. Renaming or retyping fails the
  gate.
- **In-process p95 latency budget** runnable on opt-in (`pytest -m
  loadtest`) so the numbers reflect framework + middleware + agent
  overhead, isolated from network jitter.
- **Locust profile** ([`loadtest/locustfile.py`](loadtest/locustfile.py))
  for real-OpenAI numbers against a deployed instance.

---

## Repository structure

```
clarion/
  agents/             Single-Agent ReAct loop + OpenAI client wrapper
  multiagent/         LangGraph backend: router, specialists, supervisor
  sentinel/           Trust engine — guardrails, judge, escalation scorer, PHI
  schemas/            Pydantic wire models — the contract layer between modules
  modules/            Opt-in post-launch capabilities
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
  routes/             /chat /voice/turn /cost/extract-invoice /health
  middleware/         Correlation IDs + token-bucket rate limiter
  sessions.py         Per-(tenant, conversation) session manager

gradio_app/
  app.py              11-tab Blocks shell + customer switcher
  views/              One file per tab — pure HTML render functions
  components.py       Shared visual primitives (KPI tile, donut, page intro)
  data_sources.py     Typed roll-ups consumed by the views
  tab_*.py            Stateful tabs (live agent, voice agent, cost OCR)
  theme.py / style.css   Design tokens and primitive CSS (~1,300 lines)

configs/              Per-tenant YAML (single source of customer behavior)
data/                 Per-tenant artifacts (gitignored — regenerated by harness)
tests/                580 pytest tests (unit + integration + e2e)
docs/                 Discovery doc, dev guide, deploy guide, security review
loadtest/             Locust profile + in-process p95 SLA test
huggingface/          HF Spaces deployment manifest
```

---

## Roadmap

| Milestone | Status |
|---|---|
| v1.0.0 release with locked evaluation schema | [shipped 2026-06-12](https://github.com/Ranjith200228/clarion/releases/tag/v1.0.0) |
| LangGraph multi-agent backend (opt-in per tenant) | shipped |
| PMS writeback module with PHI redaction | shipped |
| XGBoost no-show prediction folded into the report | shipped |
| Voice layer (Whisper + OpenAI TTS) | shipped |
| Production hardening (retry, breaker, rate limit, structured logs) | shipped |
| Invoice OCR via gpt-4o-mini Vision | shipped |
| Three-layer patient detail validation | shipped |
| **Next**: live-mode evaluation with real OpenAI numbers in the report | in progress |
| **Next**: fine-tuned classifier behind `BookingSpecialist` to cut prompt size and latency | planned |
| **Next**: custom domain in front of the Hugging Face Space | planned |

---

## Reading deeper

| Document | What's in it |
|---|---|
| [docs/discovery.md](docs/discovery.md) | Customer-problem framing, requirements, and the architecture's mapping onto them. |
| [docs/developer_guide.md](docs/developer_guide.md) | Local setup, internal layout, and the six-step new-tenant onboarding flow. |
| [docs/deployment_guide.md](docs/deployment_guide.md) | Container build, secret management, and the four supported hosting targets. |
| [docs/security_review.md](docs/security_review.md) | STRIDE-shaped audit including the gaps list to HIPAA. |
| [reports/v1.0.0/](reports/v1.0.0/) | Locked-schema evaluation reports for both shipped tenants. |
| [CHANGELOG.md](CHANGELOG.md) | What shipped, in order, with code pointers. |

---

[MIT](LICENSE) &copy; 2026 Ranjith Maddirala.

I'm always interested in discussing the design decisions in this
project &mdash; particularly the trust engine, the schema-locked
contract, and the multi-agent tool-scoping. Reach me at
**ranjithmaddirala24@gmail.com**.
