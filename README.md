<div align="center">

# Clarion

**Configurable multi-agent voice + vision platform for healthcare operations.**

Built end-to-end: a multi-tenant control plane that runs a LangGraph
agent over a typed tool registry, scores every reply with an
independent trust engine, and surfaces the result through an 11-tab
operations dashboard with voice in/out and invoice OCR.

[![Tests](https://img.shields.io/badge/tests-580%20passing-22D3EE?style=flat-square)](#testing--quality-gates)
[![Coverage](https://img.shields.io/badge/ruff-clean-22D3EE?style=flat-square)](#testing--quality-gates)
[![Python](https://img.shields.io/badge/python-3.11%2B-22D3EE?style=flat-square)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-22D3EE?style=flat-square)](LICENSE)
[![HF Space](https://img.shields.io/badge/demo-Hugging%20Face%20Space-FFD21E?style=flat-square)](huggingface/README.md)

[**Live demo**](huggingface/README.md) &middot;
[**Architecture**](#architecture) &middot;
[**Quick start**](#quick-start) &middot;
[**Roadmap**](#roadmap)

</div>

---

## What it is, in 60 seconds

> A real **front-desk operations platform** for specialty medical
> practices &mdash; not a chatbot demo. The agent books appointments,
> handles eligibility, recognises emergencies, and escalates to a
> human at the right moment. Every reply is scored by an independent
> trust engine before it leaves the system. Onboarding a new vertical
> is a **one-day YAML change**, not an engineering project.

| Surface | What it does |
|---|---|
| **Mission Control** | Cross-tenant operational health rolled up to a single glance. *"If it matters, it surfaces here."* |
| **Sentinel Operations** | Per-trace verdicts from the trust engine &mdash; what passed, what was caught, why. |
| **Agent Flow** | Live trace through the multi-agent LangGraph that handled the conversation. |
| **Voice Intelligence** | Frustration, emotion, and escalation signals lifted from voice transcripts. |
| **Patient 360** | Every fact this tenant knows about a patient &mdash; chart, history, payer, care team, appointments. |
| **Cost & SLO + Invoice OCR** | Per-tenant spend, latency, SLA evidence &mdash; plus a gpt-4o-mini vision uploader that lifts line-items off any vendor invoice. |
| **Healthcare Ops** | Bookings, no-shows, escalations, revenue recovered &mdash; the domain rollup. |
| **Live Agent** | Chat with the same LangGraph agent your operators trust in production. |
| **Voice Agent** | Voice in, voice out &mdash; end-to-end conversation with the agent. |
| **System Health** | Every subsystem's status, latency, and last heartbeat. |
| **Configuration** | How this tenant is wired &mdash; identity, tools, escalation thresholds, persona. |

---

## Screenshots

| | |
|:--:|:--:|
| ![Sentinel Operations](docs/screenshots/03-sentinel-ops.png) <br/> **Sentinel Operations** &mdash; composite trust gauge + signal breakdown across the last 100 turns. The trust engine grades every agent reply independently. | ![Agent Flow](docs/screenshots/04-agent-flow.png) <br/> **Agent Flow** &mdash; live trace through the multi-agent LangGraph: router &rarr; booking specialist &rarr; tools &rarr; Sentinel, with the chosen path highlighted. |
| ![Voice Intelligence](docs/screenshots/05-voice-intelligence.png) <br/> **Voice Intelligence** &mdash; emotion donut + frustration trace + escalation prediction lifted from voice transcripts. | ![Patient 360](docs/screenshots/06-patient-360.png) <br/> **Patient 360** &mdash; roster chips + profile card with engagement / sentiment / trust scores; care team + insurance below. |
| ![Cost & SLO](docs/screenshots/07-cost-slo.png) <br/> **Cost & SLO** &mdash; per-tenant spend + latency + cost-share donut. The financial control plane for the platform. | ![Invoice OCR](docs/screenshots/07-cost-slo-ocr.png) <br/> **Invoice OCR &middot; gpt-4o-mini vision** &mdash; drop a vendor invoice image, get every dollar amount lifted into a structured row with running total. |
| ![Live Agent](docs/screenshots/08-live-agent.png) <br/> **Live Agent** &mdash; chat with the production LangGraph agent. Tool calls + cost + escalation score surfaced after every turn. | ![Voice Agent](docs/screenshots/09-voice-agent.png) <br/> **Voice Agent** &mdash; voice in, voice out. Tap record, speak, tap stop, the reply plays back automatically. |

---

## Why this matters to a hiring manager

| What you'll see | Why it matters |
|---|---|
| **Production engineering**, not a notebook | Multi-stage Dockerfile, structured JSON logging, correlation IDs, retry with backoff, circuit breaker around the LLM, per-tenant rate limits, load-tested p95 SLA, [security review](docs/security_review.md). |
| **Type-safe, tested, lintable** | Pydantic v2 schemas at every boundary, mypy strict, ruff clean, **580 tests passing**. Boundary regex guards on `patient_id`/`patient_name`/`patient_phone`/`patient_email` &mdash; live-incident-driven, see commit [`a36a72f`](https://github.com/Ranjith200228/clarion/commit/a36a72f). |
| **Multi-modal AI integration** | OpenAI gpt-4o-mini for chat, Whisper-1 for STT, tts-1 for TTS, gpt-4o-mini vision for invoice OCR. Adapter Protocols so any of the four can be swapped per deployment. |
| **Multi-agent orchestration** | LangGraph hierarchical graph: classifier router &rarr; 5 specialists (Booking, Eligibility, Info, Cancel, Emergency) &rarr; supervisor that decides finish/route/escalate. Per-specialist tool advertisement shrinks prompt-injection blast radius. |
| **Independent trust engine** | Sentinel: guardrails (pattern), LLM-as-Judge (post-hoc), escalation scorer (5 weighted signals). Each component has a deliberate failure mode &mdash; guardrails prefer false alarms; the judge defaults to low-confidence on malformed JSON; the scorer is tunable per-customer. |
| **Schema-locked evaluation** | 100-scenario synthetic corpus per tenant. Locked `EvaluationReport` v1.0.0 contract. The dashboard imports `clarion.schemas` only &mdash; structurally impossible to recompute a metric inside the UI. |
| **Honest scope** | Synthetic non-PHI data. Healthcare is the *demonstration vertical*; the platform is config-driven. [Security review](docs/security_review.md) names the gaps to HIPAA explicitly. |

---

## Architecture

```
                 ┌─────────────────────────────────────────┐
                 │     11-tab Gradio dashboard (UI)        │
                 │  reads only clarion.schemas (typed)     │
                 └─────────────┬───────────────────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
            ▼                                     ▼
 ┌────────────────────┐              ┌────────────────────────┐
 │  FastAPI service   │              │  Evaluation harness    │
 │  /chat /voice/turn │              │  100 scenarios/tenant  │
 │  /cost/extract-…   │              │  locked v1.0.0 schema  │
 └─────────┬──────────┘              └───────────┬────────────┘
           │                                     │
           ▼                                     │
 ┌──────────────────────┐                        │
 │  Sentinel trust      │                        │
 │  guardrails + judge  │                        │
 │  + escalation scorer │◀───────────────────────┘
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
 │  SQLite store · tools│
 └──────────────────────┘
```

Per-tenant YAML (`configs/<customer>.yaml`) drives identity, enabled
tools, escalation thresholds, RAG corpus, and agent persona. Agent
code never branches on `customer_id`; multi-tenancy is **an allowlist,
not a code branch**.

See [docs/architecture.png](docs/architecture.png) for the full
component diagram; [docs/architecture.mmd](docs/architecture.mmd) for
the Mermaid source.

---

## Tech stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.11+ |
| **Types & validation** | Pydantic v2, mypy `--strict`, ruff |
| **LLM** | OpenAI `gpt-4o-mini` (chat + vision), `whisper-1` (STT), `tts-1` (TTS), `text-embedding-3-small` (RAG) |
| **Multi-agent** | LangGraph (StateGraph) |
| **API** | FastAPI, Pydantic models, ASGI |
| **UI** | Gradio 4.44 (Blocks API + custom CSS) |
| **RAG** | FAISS index + scikit-learn TF-IDF fallback |
| **ML** | XGBoost (no-show classifier with held-out ROC-AUC + top-decile lift) |
| **Data store** | SQLite per-tenant (`structured.sqlite`) for slots, appointments, eligibility, PMS tasks |
| **Observability** | Structured JSON logs, correlation IDs, span traces (JSONL), audit log per tenant |
| **Resilience** | Retry-with-backoff, circuit breaker around LLM, per-(customer, IP) token-bucket rate limit |
| **Deploy** | Multi-stage Docker, Hugging Face Spaces (primary), Cloud Run / Render / Fly.io manifests, Poetry |
| **Test & CI** | 580 pytest tests, GitHub Actions matrix on 3.11 + 3.12, pre-commit hooks, security review (STRIDE) |

---

## Quick start

```bash
# 1. Clone + install
git clone https://github.com/Ranjith200228/clarion.git
cd clarion
poetry install

# 2. Set the LLM key
export OPENAI_API_KEY=sk-...

# 3. Populate evaluation data (synthetic, one-time)
poetry run python -m clarion.eval --customer all

# 4. Run the FastAPI backend (terminal 1)
poetry run python -m uvicorn api.app:app --host 0.0.0.0 --port 8000

# 5. Run the Gradio dashboard (terminal 2)
poetry run python -m gradio_app

# 6. Open http://localhost:7860
```

The dashboard works without the FastAPI backend &mdash; only the
**Live Agent**, **Voice Agent**, and **Invoice OCR** tabs need
it. Everything else reads JSON + SQLite artifacts from disk.

---

## Key features

### 1. Configurable multi-tenancy

Customer-specific behavior is a YAML field, not a code branch:

```yaml
# configs/ophthalmology.yaml
customer_id: ophthalmology
display_name: North Shore Eye Associates
enabled_tools:
  - search_slots
  - book_appointment
  - cancel_appointment       # orthopedics drops this; cancels route to a PMS task
  - check_eligibility
  - create_pms_task
escalation:
  low_confidence: 0.60
  frustration: 0.70
  max_clarifications: 3
use_multiagent: true         # opt into LangGraph backend
```

Adding a new vertical is six steps (one working day), none of which
touch `clarion/agents/`. See [docs/developer_guide.md](docs/developer_guide.md).

### 2. Schema-validated agent tools

Every LLM tool call is Pydantic-validated **before** it touches the
store. Real incident from the audit log:

> The LLM hallucinated a patient's full name into the `patient_id`
> field on `book_appointment`. A row landed in the structured store
> with `patient_id = "Ranjit Kumar Madhirala"`, corrupting downstream
> Patient 360 renders.

The fix &mdash; commit [`a36a72f`](https://github.com/Ranjith200228/clarion/commit/a36a72f),
visible in [`clarion/schemas/tools.py`](clarion/schemas/tools.py)
&mdash; adds three regex guards:

```python
patient_id    : Field(pattern=r"^[A-Za-z][A-Za-z0-9_\-]{0,63}$")
patient_name  : Field(pattern=r"^[A-Za-zÀ-ÿ'\-]{2,}(\s+[A-Za-zÀ-ÿ'\-]{1,}){1,4}$")
patient_phone : Field(pattern=r"^[\d\s\-\(\)\+\.]{7,25}$")
patient_email : Field(pattern=r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")
```

Combined with the booking-specialist persona prompt ("read each
field back to the caller before booking"), the agent now collects
real contact details at every touchpoint. The Patient 360
confirmation card surfaces the actually-collected values, not
synthesised cosmetics.

### 3. Sentinel trust engine

Three independent components, each scored per turn:

| Component | Output | Failure mode |
|---|---|---|
| **Guardrails** (emergency / clinical / PHI) | Short-circuit reply &mdash; LLM is never called | Pattern-based: prefer false alarms |
| **LLM-as-Judge** (booking + hallucination + policy) | Structured verdict per turn | Defensive parsing: low-confidence on malformed JSON |
| **Escalation scorer** (5 weighted signals) | 0&ndash;1 score per turn | Tunable thresholds per customer YAML |

Independence is the point: even if the agent's reply was correct,
the trust engine grades it from outside. The Sentinel Operations
tab visualises every verdict.

### 4. Invoice OCR via gpt-4o-mini Vision

End-to-end multimodal pipeline shipped in [`f336fa2`](https://github.com/Ranjith200228/clarion/commit/f336fa2):

```
Gradio upload  ─▶  FastAPI /cost/extract-invoice  ─▶  OpenAI Vision
  (image bytes)     (multipart + 5MB cap)             (gpt-4o-mini)
       ◀─────────────────────◀───────────────────────────◀
   Styled card     Structured JSON: line items, total, currency
```

JSON-mode response, defensive parsing (strips code fences, coerces
number-as-string), per-row rendering with a highlighted total + raw
text drawer for verification.

### 5. Engagement layer that signals product taste

Five micro-features that make the dashboard feel alive without
adding meaningful runtime cost:

- **Time-of-day greeting**: *"Good evening, operator. Ophthalmology &middot; 3 items need your eyes"*
- **Today's Standout card** that names the one thing that matters right now
- **Per-tenant accent identity** (ophthalmology = cyan, orthopedics = amber) that ripples through KPI edges, badges, focus rings
- **KPI value entrance animation** &mdash; values scale+fade in on every customer switch
- **Logo easter egg** &mdash; click the hex mark for a 700ms spin with a cyan glow burst

All five honor `prefers-reduced-motion` and are pure-CSS where possible.

---

## Repository structure

```
clarion/
  agents/             Agent core: ReAct loop + tool dispatch + OpenAI client
  multiagent/         LangGraph backend: router + specialists + supervisor
  sentinel/           Trust engine: guardrails, judge, escalation, PHI
  schemas/            Pydantic wire models (the lock between layers)
  modules/            Opt-in post-launch modules
    invoice_ocr.py        gpt-4o-mini vision (shipped)
    no_show_prediction/   XGBoost no-show classifier (shipped)
    pms_writeback/        Conversation -> summary.json + task.json (shipped)
    voice/                STT + TTS + VoiceOrchestrator (shipped)
  pipelines/          RAG + structured store
  resilience/         Retry, circuit breaker, rate limit
  evaluation/         100-scenario harness + locked report writer
  observability/      Structured logging + correlation IDs + spans
  config/             Settings + per-customer YAML loader

api/
  app.py              FastAPI factory (modular routers)
  routes/             /chat /voice/turn /cost/extract-invoice /health /evaluate
  middleware/         Correlation IDs + rate limiter
  sessions.py         Per-(customer, conversation) session manager

gradio_app/
  app.py              11-tab Blocks shell, customer switcher, identity layer
  views/              One file per tab: pure HTML renderers, never any I/O
  components.py       Shared visual primitives (kpi_tile, donut, page_intro, …)
  data_sources.py     Typed roll-ups consumed by views
  tab_*.py            Stateful tabs (live agent, voice agent, cost OCR)
  theme.py + style.css   ~1300 lines of design tokens + primitives

configs/              Per-customer YAML (ophthalmology, orthopedics)
data/                 Per-customer artifacts (report + trace JSONs, SQLite)
tests/                580 pytest tests (unit + integration + e2e)
docs/                 Architecture diagrams, dev + deploy guides, security review
loadtest/             Locust profile + in-process p95 SLA test
huggingface/          HF Spaces deployment manifest
```

---

## Testing & quality gates

```bash
poetry run pytest                    # 580 tests, ~30s
poetry run ruff check clarion gradio_app api tests
poetry run mypy --strict clarion api
poetry run pytest -m loadtest        # opt-in p95 SLA budget
```

CI matrix on Python 3.11 + 3.12 (GitHub Actions). Every commit:
- Pytest (must be green)
- Ruff lint (no autofix at gate)
- Mypy strict on core
- Schema regression test on the locked `EvaluationReport` v1.0.0

The boundary regex guards mentioned above ([commit `a36a72f`](https://github.com/Ranjith200228/clarion/commit/a36a72f))
ship with 6 new unit tests that lock in the rejection of "single-word
name", "phone with words", "n/a email", plus the acceptance of
international names + E.164 formats.

---

## Deployment

| Target | Manifest | Notes |
|---|---|---|
| **Hugging Face Space** (primary) | [`huggingface/README.md`](huggingface/README.md) | Docker SDK, `app_port: 7860` |
| GCP Cloud Run | [`deploy/cloudrun.yaml`](deploy/cloudrun.yaml) | Knative + Secret Manager |
| Render | [`deploy/render.yaml`](deploy/render.yaml) | Blueprint, `sync: false` secret |
| Fly.io | [`deploy/fly.toml`](deploy/fly.toml) | `fly secrets set OPENAI_API_KEY=...` |
| Local | `docker compose up` | API + Gradio side-by-side |

Multi-stage Dockerfile pre-bakes FAISS indices in the builder stage
so container start is <2 s. Non-root user (UID 1000), HEALTHCHECK on
`/health`, tini for signal propagation. See
[docs/deployment_guide.md](docs/deployment_guide.md).

---

## Results

100 scenarios per customer, scripted-mode evaluation:

| Metric | Ophthalmology | Orthopedics |
|---|---|---|
| Pass rate | 100% | 100% |
| Containment rate | 0.74 | 0.74 |
| Booking accuracy | 1.00 (38/38) | 1.00 (38/38) |
| Escalation precision | 0.62 | 0.62 |
| Escalation recall | 1.00 | 1.00 |
| Safety catch rate (emergencies) | 1.00 | 1.00 |

100% recall on emergencies is the spec-mandated floor; we hit it.
Full reports at [`reports/v1.0.0/`](reports/v1.0.0/).

---

## Roadmap

| Module | Status |
|---|---|
| M1 &mdash; PMS Writeback (conversation &rarr; `summary.json` + `task.json`) | shipped |
| M3 &mdash; No-Show Prediction (XGBoost, held-out ROC-AUC + top-decile lift) | shipped |
| M5 &mdash; Voice Layer (faster-whisper STT + OpenAI TTS) | shipped |
| LangGraph multi-agent backend (router &rarr; specialists &rarr; supervisor) | shipped |
| Phase 18 hardening (retries, breaker, rate limit, JSON logs, load test) | shipped |
| v1.0.0 release with locked schema | [released 2026-06-12](https://github.com/Ranjith200228/clarion/releases/tag/v1.0.0) |
| Cost & SLO invoice OCR (gpt-4o-mini vision) | shipped |
| Engagement layer + per-tab page intros + boundary validators | shipped |
| Patient detail capture in conversation &rarr; confirmation card | shipped |
| **Next**: live-mode evaluation report with real OpenAI numbers | in progress |
| **Next**: fine-tuned classifier behind `BookingSpecialist` | planned |

---

## Read deeper

| Document | What's inside |
|---|---|
| [`docs/discovery.md`](docs/discovery.md) | The FDE artifact: customer problem &rarr; requirements &rarr; how Clarion's three layers map onto them. |
| [`docs/developer_guide.md`](docs/developer_guide.md) | Local setup, six-step new-customer onboarding, internal layout. |
| [`docs/deployment_guide.md`](docs/deployment_guide.md) | Container build, secret management, the four hosting targets in detail. |
| [`docs/security_review.md`](docs/security_review.md) | STRIDE-shaped audit. The gaps list is the load-bearing section. |
| [`reports/v1.0.0/`](reports/v1.0.0/) | Locked-schema evaluation reports for both shipped customers. |
| [`CHANGELOG.md`](CHANGELOG.md) | What shipped, with pointers into the code. |

---

## License & contact

[MIT](LICENSE) &copy; 2026 Ranjith Maddirala.

Built end-to-end as a portfolio piece. Reach out at
`ranjithmaddirala24@gmail.com` &mdash; happy to walk through any
layer in detail.
