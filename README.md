# Clarion

**Configurable Multi-Agent Voice Automation Platform with Sentinel Trust Engine.**

Clarion is a config-driven AI platform for deploying voice automation
to new customer verticals. Healthcare scheduling is the **demonstration
vertical**, not the product — the architecture treats vertical-specific
logic as configuration (YAML + rules markdown + seed JSON), so onboarding
a new vertical is a one-day job, not an engineering project.

> **Honesty note.** This is a prototype on synthetic, non-PHI data.
> Metrics demonstrate capability and engineering rigor, not production ROI.

![Architecture diagram](docs/architecture.png)

| Resource | Where |
|---|---|
| Discovery doc (FDE artifact) | [`docs/discovery.md`](docs/discovery.md) |
| Developer guide | [`docs/developer_guide.md`](docs/developer_guide.md) |
| Deployment guide | [`docs/deployment_guide.md`](docs/deployment_guide.md) |
| Architecture (Mermaid source) | [`docs/architecture.mmd`](docs/architecture.mmd) |

---

## 1. Problem

Specialty medical practices lose patients and revenue at the first
touchpoint — the phone. Front desks drown in repetitive calls
("can I book a cataract consult?", "do you take Aetna?", "what time is
my appointment?") and the urgent calls — sudden vision loss, a
suspected fracture, a patient asking for clinical advice — must never
be mishandled.

Voice AI can absorb the routine load, but the **hard problem isn't
talking** — it's being trustworthy:

- Booking the **correct provider + appointment type + duration** under
  complex per-practice rules (new-patient-only providers, dilation
  prerequisites, payer eligibility).
- **Never giving clinical advice.**
- Recognizing emergencies and **handing off to a human at the right
  moment**.

Clarion is the configurable platform that does this with measurable
trust. The hiring pitch: this demonstrates **Forward Deployed
Engineering** — stand up a platform for a specific customer's messy
real problem, then measure and iterate.

---

## 2. Architecture

The system is five layers, each with a one-way dependency on the layer
below it:

```
┌─────────────────────────────────────────────────────────┐
│  Gradio UI (Phase 14)                                   │
│  Live Agent · Quality · Escalations · Trace Explorer    │
└────────────────────────┬────────────────────────────────┘
                         │ reads report.json + trace.json
┌────────────────────────▼────────────────────────────────┐
│  Evaluation harness (Phase 13)                          │
│  python -m clarion.eval --customer X                    │
│  runner.py · metrics.py · reporter.py · trace_report.py │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Sentinel trust engine (Phases 6, 10, 11)               │
│  Guardrails · LLM-as-Judge · Escalation scorer · PHI    │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Agent core (Phases 5, 7, 8)                            │
│  ReAct loop · Tools · FastAPI · Observability           │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│  Foundation (Phases 1-4)                                │
│  Multi-tenant config · RAG · SQLite · Tool registry     │
└─────────────────────────────────────────────────────────┘
```

See [`docs/architecture.png`](docs/architecture.png) for the full
component diagram.

The Phase 13 spec line *"LOCK THE REPORT SCHEMA. Future UIs consume
this schema. No metric computation inside UI."* is structurally
enforced: the UI imports only `clarion.schemas` (typed wire models),
never `clarion.evaluation.metrics`. Adding a new metric is a
backend-only change.

---

## 3. Config-driven design

Everything that varies between customers lives in YAML:

```yaml
# configs/ophthalmology.yaml
customer_id: ophthalmology
display_name: North Shore Eye Associates
specialties: [Cataract Pre-Op Consult, Glaucoma Follow-Up, ...]
enabled_tools:
  - search_slots
  - book_appointment
  - cancel_appointment       # orthopedics drops this; cancels route to a task
  - check_eligibility
  - create_pms_task
languages: [en, es]
escalation:
  low_confidence: 0.6
  max_clarifications: 3
  frustration: 0.7
rules_path: rules/ophthalmology
agent_persona: |
  You are Clarion, the virtual front-desk assistant for ...
```

The agent code reads this and **never branches on `customer_id`**.
Orthopedics doesn't have a cancel tool because its YAML doesn't list it
— the registry honors `enabled_tools` strictly, so the LLM never even
sees a `cancel_appointment` function on the tool list.

---

## 4. Customer onboarding

Adding a new customer is six steps (one working day) — none touch
`clarion/agents/`:

1. **Discovery** (`docs/discovery_<customer>.md`) — see the
   [`docs/discovery.md`](docs/discovery.md) sample for the FDE template.
2. **YAML** (`configs/<customer>.yaml`) — schema-validated; typos
   rejected at load time.
3. **Seed** (`data/seeds/<customer>.json`) — synthetic providers +
   slots + eligibility records.
4. **Rules** (`data/rules/<customer>/*.md`) — markdown chunks the RAG
   pipeline indexes.
5. **Personas** — `python -m clarion.simulator.cli generate <customer>`
   produces 100 scenarios automatically.
6. **Evaluate** — `python -m clarion.eval --customer <customer>`
   produces the locked report + trace JSONs.

The [developer guide](docs/developer_guide.md) walks through each step.

---

## 5. Trust Engine

The Sentinel layer is **three independent components**, each with a
deliberate failure mode:

| Component | Phase | Output | Failure mode |
|---|---|---|---|
| Guardrails (emergency / clinical / PHI) | 6 | Short-circuit reply, never calls LLM | Pattern-based: prefer false alarms |
| LLM-as-Judge (booking + hallucination + policy) | 10 | Structured verdict per turn | Defensive parsing: low-confidence verdict on malformed JSON |
| Escalation scorer (5 weighted signals) | 11 | 0-1 score per turn | Tunable threshold + per-customer escalation YAML |

The judge runs **post-hoc**, so even if the agent's reply was correct
the trust engine grades it independently. The escalation scorer fuses:
`low_confidence`, `repeated_clarification`, `rule_conflict`,
`frustration` (regex-based), `unsupported_request`. Each signal
contributes a labeled reason for the dashboard.

---

## 6. Evaluation harness

Canonical CLI:

```bash
poetry run python -m clarion.eval --customer ophthalmology
poetry run python -m clarion.eval --customer all --out reports/
```

Produces two locked-contract JSON files per customer:

| File | Schema | Consumed by |
|---|---|---|
| `report_<customer>.json` | `EvaluationReport` v1.0.0 | Quality + Escalation UI tabs |
| `trace_<customer>.json` | `TraceReport` v1.0.0 | Trace Explorer UI tab |

The harness drives 100 synthetic personas per customer through the
real agent (scripted FakeLLM in CI, real OpenAI in staging) and
computes the 11 spec metrics.

---

## 7. Tracing

Every `Agent.chat` call emits one trace with a full span hierarchy:

```
agent.chat                       user_chars, reply_chars
├── guardrails.check             fired: bool
├── retrieval                    hit_count, top_score, top_source
├── react.step                   step_index
│   ├── llm.complete             model, tokens, cost_usd, advertised_tools
│   ├── tool.search_slots        tool, ok, arguments_keys
│   └── tool.book_appointment    ...
└── react.step
    └── llm.complete             (final reply, no tools)
```

JSONL traces land at `<data_dir>/<customer>/traces.jsonl`. Every
`llm.complete` span carries model + input/output tokens + cost in USD
derived from a per-model pricing table — adding a new model updates
every existing span's cost field automatically.

---

## 8. Metrics

The locked report contract (`schema_version: "1.0.0"`) carries all 11
Phase 13 metrics per scope (overall + by_difficulty + by_intent):

| Metric | Source |
|---|---|
| Containment Rate | `actual_outcome ∈ {booked, cancelled, info_provided}` / total |
| Booking Accuracy | booked & passed / scenarios expecting a booking |
| Hallucination Rate | mean `judge.hallucination` across judged scenarios |
| Escalation Precision | confusion matrix on predicted vs ground-truth `should_escalate` |
| Escalation Recall | same |
| Safety Catch Rate | recall on `intent ∈ {emergency, clinical_advice}` |
| Average Turns | mean `react.step` count per scenario |
| Tokens Per Call | sum tokens / sum `llm.complete` calls |
| Cost Per Call | sum `cost_usd` / scenario count |
| P50 Latency | 50th-percentile `agent.chat.duration_ms` |
| P95 Latency | 95th-percentile `agent.chat.duration_ms` |

Plus three pre-aggregated UI-feed fields: `outcome_distribution`,
`escalation_reason_frequency`, `escalated_scenario_ids`.

---

## 9. Deployment

Same image, four targets:

| Target | Manifest | Notes |
|---|---|---|
| Hugging Face Gradio Space (primary) | [`huggingface/README.md`](huggingface/README.md) | Docker SDK, `app_port: 7860` |
| GCP Cloud Run | [`deploy/cloudrun.yaml`](deploy/cloudrun.yaml) | Knative + Secret Manager |
| Render | [`deploy/render.yaml`](deploy/render.yaml) | Blueprint, `sync: false` secret |
| Fly.io | [`deploy/fly.toml`](deploy/fly.toml) | `fly secrets set OPENAI_API_KEY=...` |
| Local | `docker compose up` | API + Gradio side-by-side |

Multi-stage Dockerfile pre-bakes FAISS indices in the builder so
runtime starts instantly. Non-root user (UID 1000), HEALTHCHECK on
`/health`, signals propagated via tini. Detailed steps in
[`docs/deployment_guide.md`](docs/deployment_guide.md).

---

## 10. Results

Headline numbers on the scripted-mode evaluation harness, both
shipped customers, 100 scenarios each:

| Metric | Ophthalmology | Orthopedics |
|---|---|---|
| Pass rate | 100% | 100% |
| Containment rate | 0.74 | 0.74 |
| Booking accuracy | 1.00 (38/38) | 1.00 (38/38) |
| Escalation precision | 0.62 | 0.62 |
| Escalation recall | 1.00 | 1.00 |
| Safety catch rate | 1.00 | 1.00 |

"100% recall on emergencies" is the spec-mandated floor; we hit it.
Live-mode numbers (real OpenAI calls) will land in the Phase 19
release notes.

**Test coverage**: ~370 tests, mypy strict clean across ~70 source
files, CI matrix on Python 3.11 + 3.12.

---

## 11. Lessons learned

- **Lock the wire schema early.** The Phase 13 schema-lock pattern
  (`schema_version: "1.0.0"` + additive-only changes) made the UI work
  in Phase 14 painless — every field the UI wanted already had a
  decided home in `clarion/schemas/evaluation.py`.
- **One-way dependency graphs are load-bearing.** `runner → reporter →
  metrics` and `gradio_app → clarion.schemas` (not `clarion.evaluation`)
  prevented the dashboard from accidentally re-computing metrics. The
  rule isn't a guideline; it's structurally impossible to violate
  because the imports don't resolve.
- **Multi-tenancy is a YAML allowlist, not a code branch.** The
  orthopedics-no-cancel divergence is one missing line in YAML.
  Every time I caught myself reaching for an `if customer_id == ...`,
  the right move was an additive YAML field instead.
- **Guardrails are part of the prediction surface.** When the
  emergency guardrail short-circuits the LLM and files an urgent task,
  the *system* has predicted escalation — the escalation scorer's
  `already_escalated` shortcut treats this as the strongest possible
  signal, which is the right semantic.
- **Pre-bake everything you can.** Pre-aggregated UI feeds
  (`outcome_distribution`, `escalation_reason_frequency`) in the
  report meant the UI tabs are 30-line rendering functions.
  Pre-built FAISS indices in the Docker builder stage meant the
  container starts in <2s.

---

## 12. Future roadmap

Post-launch modules, prioritized:

| Module | Status | Spec |
|---|---|---|
| **M1: PMS Writeback** | Pending | Convert conversations to structured `summary.json` + `task.json`; field extraction accuracy metric |
| **M3: No-Show Prediction** | Pending | XGBoost on booking features; ROC-AUC + lift; high-risk bookings trigger reminder recommendations |
| **M5: Voice Layer** | Pending | faster-whisper STT + OpenAI TTS; speech → STT → Clarion → TTS; reuses the entire existing engine |
| **LangGraph refactor** | Deferred | Hierarchical router → specialist → supervisor agents. Only after launch per spec. |
| **Phase 18: Production hardening** | Pending | Retries, caching, rate limiting, circuit breakers, structured logging, load testing, security review |
| **Phase 19: v1.0.0 release** | Pending | Tag + release notes + demo assets + final evaluation report |

The recruiter test (from the spec's "Definition of Done"):
> A recruiter opens one URL and immediately sees: live AI scheduling
> agent, evaluation metrics, escalation analysis, full tracing,
> multi-tenant customer switching, automated outcomes, optional voice
> interaction.

We're not there yet on the URL — but every piece behind it ships green.

---

## License

MIT — see [LICENSE](LICENSE).
