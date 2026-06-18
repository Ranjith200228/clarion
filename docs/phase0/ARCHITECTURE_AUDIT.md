# ARCHITECTURE_AUDIT.md — Clarion as it exists today

*Phase 0 deliverable. Read-only inventory. Nothing here is changing
yet; the next document (`UI_GAP_ANALYSIS.md`) is where we name what's
missing, and `IMPLEMENTATION_PLAN.md` is where we describe how the
experience layer gets rebuilt on top of this stack.*

---

## 1. Repository structure

```
clarion/                            (~10,200 LoC source, 555 tests)
├── api/                            FastAPI service
│   ├── app.py                      create_app factory; mounts routers, middleware, voice orchestrator
│   ├── sessions.py                 in-process Session + per-customer resource cache; LLM factory with DemoMode fallback
│   ├── middleware/
│   │   ├── correlation.py          X-Request-Id contextvar bind
│   │   └── rate_limit.py           per-(customer, ip) token bucket
│   ├── routes/
│   │   ├── chat.py                 POST /chat (soft-error guard wraps the agent call)
│   │   ├── voice.py                POST /voice/turn (base64 audio in/out, length-mismatch detection)
│   │   ├── evaluate.py             POST /evaluate (kicks off scripted harness)
│   │   └── health.py               GET /health
│   └── schemas.py                  request/response Pydantic models
│
├── clarion/                        engine
│   ├── agents/                     single-Agent ReAct backend
│   │   ├── agent.py                Agent class; rolling transcript, trace root, audit hooks
│   │   ├── react.py                react_loop, ToolDispatcher
│   │   ├── llm.py                  LLMClient Protocol + FakeLLM + Message/ToolCall types
│   │   ├── openai_client.py        OpenAIClient (retry-wrapped network boundary)
│   │   └── prompt.py               system prompt assembly w/ persona + retrieval
│   ├── multiagent/                 LangGraph backend (opt-in via CustomerConfig.use_multiagent)
│   │   ├── state.py                MultiAgentState TypedDict, SpecialistIntent, SupervisorDecision
│   │   ├── router.py               LLMIntentRouter + HeuristicIntentRouter (5-way classifier)
│   │   ├── specialists/            Booking, Eligibility, Info, Cancel, Emergency (tool-scoped)
│   │   ├── supervisor.py           rule-based finish/route/escalate decision tree
│   │   └── runner.py               StateGraph assembly + Agent-compatible chat(str)->str
│   ├── sentinel/                   trust engine
│   │   ├── guardrails.py           regex guardrails for emergencies + clinical advice
│   │   ├── judge.py                LLM-as-judge (hallucination + booking-correctness + policy-violation)
│   │   ├── escalation.py           composite EscalationScorer with 5 weighted signals
│   │   ├── frustration.py          turn-over-turn frustration detection
│   │   ├── phi.py                  regex-based PHI redactor (phones, member ids, emails, SSNs, patient ids)
│   │   └── audit.py                AuditLog writer (PHI redacted at boundary)
│   ├── observability/              spans + cost + structured logging
│   │   ├── tracer.py               Span / Tracer / Trace; nested span hierarchies
│   │   ├── writer.py               TraceWriter (JSONL per customer)
│   │   ├── cost.py                 per-model pricing table; cost_usd()
│   │   └── logging.py              JsonFormatter + correlation_id_scope contextvar
│   ├── rag/                        retrieval
│   │   ├── embeddings.py           TfidfEmbedder + OpenAIEmbedder; pick_embedder()
│   │   ├── retriever.py            FAISS IndexFlatIP + per-instance LRU cache
│   │   └── builder.py              build_index() + load_customer_retriever()
│   ├── tools/                      agent tools (5 ship)
│   │   ├── search_slots.py / book_appointment.py / cancel_appointment.py
│   │   ├── check_eligibility.py / create_pms_task.py
│   │   ├── registry.py             enforces customer.enabled_tools
│   │   └── base.py                 Tool Protocol, ToolContext, ToolError
│   ├── pipelines/
│   │   ├── structured/             SQLite-backed Provider/Slot/Eligibility store
│   │   └── unstructured/           Markdown chunker for rules corpora
│   ├── modules/                    post-launch modules (opt-in per customer)
│   │   ├── pms_writeback/          M1: summary.json + task.json + PHI-redacted, field_extraction_accuracy
│   │   ├── no_show_prediction/     M3: synthetic dataset + XGBoost + risk bands + ROC-AUC + top-decile lift
│   │   └── voice/                  M5: VoiceOrchestrator + Transcriber+Speaker Protocols
│   ├── resilience/                 production hardening
│   │   ├── retry.py                full-jitter exponential backoff
│   │   └── circuit_breaker.py      closed/open/half-open around the LLM client
│   ├── schemas/                    Pydantic wire models (LOCKED at schema_version=1.0.0)
│   ├── evaluation/                 Phase 13 harness + reporter
│   ├── simulator/                  100-scenario corpus driver
│   └── config/                     CustomerConfig + Settings + YAML loader
│
├── gradio_app/                     Phase 14 product UI
│   ├── app.py                      Blocks layout; 5 tabs + customer switcher; refresh_all fan-out
│   ├── tab_live_agent.py           ChatInterface backed by /chat
│   ├── tab_voice_agent.py          gr.Microphone → /voice/turn → autoplay gr.Audio
│   ├── tab_quality.py              renders EvaluationReport.headline + outcome distribution
│   ├── tab_escalations.py          renders escalation_reason_frequency + escalated_scenario_ids
│   ├── tab_trace_explorer.py       renders TraceReport.entries as gr.Dataframe
│   ├── agent_client.py             httpx wrapper over /chat
│   ├── voice_client.py             httpx wrapper over /voice/turn
│   └── data.py                     loads report_<customer>.json + trace_<customer>.json
│
├── configs/                        per-tenant YAML
│   ├── ophthalmology.yaml
│   └── orthopedics.yaml
├── data/                           pre-built FAISS indices, structured stores, persona corpora
│   ├── personas/{ophthalmology,orthopedics}.json  (100 scenarios each)
│   ├── seeds/                      Provider/Slot/Eligibility seed JSON
│   └── rules/                      per-tenant markdown rule corpora
├── reports/v1.0.0/                 locked-schema artifacts shipped at the tag
├── loadtest/                       locust profile + pytest p50/p95 SLA marker
├── tests/                          61 test files (555 tests, mypy strict, ruff clean)
├── docs/                           discovery, developer guide, deployment, security review,
│                                   release notes, architecture mermaid
├── huggingface/                    Space metadata + README frontmatter source
├── scripts/                        build_indices.sh + serve_all.sh (single-container entrypoint)
├── Dockerfile                      builder → test → runtime; non-root UID 1000
└── docker-compose.yml              two-container dev layout
```

**Versioning.** Marketing version on `main` is `1.1.0-dev`. The tagged
v1.0.0 release is locked. Wire schemas
(`EvaluationReport`, `TraceReport`, `ConversationSummary`,
`PmsTaskWriteback`, `NoShowPrediction`, `VoiceTurnRequest`/`Response`)
are all pinned at `schema_version="1.0.0"` and the dashboard must
treat them as immutable.

---

## 2. Current UI structure (Gradio Blocks)

`gradio_app/app.py::build_app()` constructs a `gr.Blocks` with:

- **Title bar** — single `gr.Markdown("# Clarion …")` line
- **Customer dropdown** — top-of-page, fans out to every tab on
  change (and once on `demo.load`) via `refresh_all(...)`
- **Five tabs** in this order:
  1. **Live Agent** — `gr.ChatInterface` against `/chat`; shows
     running cost + tokens
  2. **Voice Agent** — `gr.Audio(sources=["microphone"])` →
     `/voice/turn` → `gr.Audio(autoplay=True)`; transcript and
     per-stage STT/agent/TTS latency below
  3. **Quality Metrics** — headline markdown + 2 dataframes
  4. **Escalations** — summary markdown + 2 dataframes
  5. **Trace Explorer** — summary markdown + 1 tall dataframe

`gradio_app/data.py` reads
`<data_dir>/<customer>/report_<customer>.json` and
`trace_<customer>.json` (the locked-schema reports). No business
logic; pure render layer.

**Theme.** Default Gradio theme. No custom CSS. No global colors.
No dark mode. No iconography.

---

## 3. FastAPI routes

| Verb + path | Owner | Purpose | Auth |
|---|---|---|---|
| `GET /health` | health.py | liveness | none |
| `POST /chat` | chat.py | one user turn; returns reply + last_turn_metrics; soft-error guard wraps `agent.chat` | none |
| `POST /voice/turn` | voice.py | STT → agent → TTS round-trip; base64 audio; n_bytes integrity check | none |
| `POST /evaluate` | evaluate.py | run the scripted harness (returns EvaluationReport) | none |

Middleware stack (LIFO, outermost first):
1. `CorrelationIdMiddleware` — X-Request-Id contextvar bind
2. `RateLimitMiddleware` — per-(customer, ip) token bucket

Settings live on `app.state`: `settings`, `sessions`, `rate_limiter`,
`voice_orchestrator` (None when no `OPENAI_API_KEY` is set).

---

## 4. LangGraph graph

Opt-in per customer via `CustomerConfig.use_multiagent: bool`
(default `False`). When enabled, `SessionManager._build_agent`
constructs a `MultiAgentRunner` instead of an `Agent`.

```
START
  ↓
router            ← IntentRouter classifies into 5 specialists
  ↓ (conditional edge on state["intent"])
{ booking | eligibility | info | cancel | emergency }
  ↓
supervisor        ← rule-based 3-way decision tree
  ↓ (conditional edge on state["decision"])
finish/escalate → END
route          → router (bounded by max_visits=3)
```

State is a `TypedDict` (`MultiAgentState`) with append-only fields
using `operator.add` reducers — set up for future parallel
specialists, even though none run in parallel today.

Specialist Protocol: each subclass declares `intent`, `allowed_tools`
(frozenset), `persona` at class scope. The base class
`Specialist.__call__` runs the existing `react_loop` with a tool-scoped
`CustomerConfig.model_copy(update={"enabled_tools": intersection})`.

`EmergencySpecialist` short-circuits without an LLM call — emits the
canned 911 reply + `escalated=True`.

---

## 5. Sentinel integrations

All entry points hooked from `clarion/agents/agent.py::Agent.chat`:

| Pre-LLM | What | Where |
|---|---|---|
| Guardrails | emergency phrase scan + clinical-advice scan | `sentinel/guardrails.py::check()` |
| PHI redaction | strips phones / SSNs / member ids before transcript persists | `sentinel/phi.py::redact()` |

| Post-LLM | What | Where |
|---|---|---|
| LLM judge | hallucination + booking-correctness + policy-violation scoring | `sentinel/judge.py::Judge.judge()` |
| Escalation scorer | 5 weighted signals → composite 0-1 + decision threshold | `sentinel/escalation.py::EscalationScorer.score()` |
| Frustration | turn-over-turn signal feeding the scorer | `sentinel/frustration.py` |
| Audit log | append-only JSONL; PHI redacted at the boundary | `sentinel/audit.py::AuditLog.write()` |

The multi-agent backend wires the same `EscalationScorer` into its
`Supervisor` node — escalation behavior stays consistent across
backends so toggling `use_multiagent` doesn't change the metric.

---

## 6. Observability integrations

`clarion/observability/`:

- **Tracer** — per-conversation; nested spans
  (`agent.chat` → `retrieval` → `react.step` →
  `llm.complete` + `tool.<name>`); attributes stamp tokens, cost
  USD, model name, escalation score
- **TraceWriter** — JSONL `<data_dir>/<customer>/traces.jsonl`
- **Cost module** — known-model pricing table; `cost_usd(model,
  input_tokens, output_tokens) -> float`
- **JSON logging** — single-line per-record formatter with
  `correlation_id` from a contextvar
- **API correlation middleware** — accepts client-supplied
  `X-Request-Id` or mints a fresh UUID; binds the contextvar so
  every log line in the request scope carries the id

`EvaluationReport` rolls these into:
- `headline` dict (six numbers for the dashboard top strip)
- `metrics.latency_ms` (avg / p50 / p95 / count)
- `metrics.cost_per_request_usd` + `tokens_per_call`

---

## 7. Retrieval

- **TfidfEmbedder** — sklearn TF-IDF + L2 normalization; default
  when no `OPENAI_API_KEY`
- **OpenAIEmbedder** — `text-embedding-3-small`; opt-in when key
  is present
- **Retriever** — FAISS `IndexFlatIP` on L2-normalized vectors
  (cosine == inner product); per-instance LRU cache with stats
- **build_index()** — `clarion/rag/builder.py`; writes
  `rules.faiss` + `rules_meta.json` + `rules_embedder.json`

The Space pre-builds indices at Docker build time via
`scripts/build_indices.sh` so first-request latency is fast.

---

## 8. Deployment flow

- **Local dev**: `docker compose up` — separate API + Gradio
  containers; no merge of stdout.
- **Single-container target** (HF Spaces, Cloud Run, Render, Fly):
  `scripts/serve_all.sh` starts FastAPI on loopback `:8000` and
  Gradio on `:7860`, forwards SIGTERM to both.
- **Image**: multi-stage; non-root UID 1000; FAISS indices pre-baked
  into `/app/data/`; size ~1.1 GB.
- **Live URL**:
  `https://huggingface.co/spaces/Ranjithmaddirala/clarion`.
- **Required secret**: `OPENAI_API_KEY` (Space falls back to demo
  mode when missing; agent reply explains how to set it).

---

## 9. Tests + quality bar

| Surface | Count |
|---|---|
| Unit + integration tests | 555 (+1 SLA under `-m loadtest`) |
| mypy strict | clean (97 source files) |
| ruff lint | clean |
| Doc-contract tests | 13 (assert README has the load-bearing
sections) |
| Locked schema versions | `EvaluationReport`, `TraceReport`,
M1/M3/M5 wire shapes all pinned at `1.0.0` |

---

## 10. What this audit explicitly leaves untouched

- **Gradio Blocks** as the UI framework
- **FastAPI** as the API framework
- **LangGraph** as the multi-agent orchestrator
- **Sentinel** modules + the same composite `EscalationScorer`
- **FAISS + embedders + TF-IDF fallback**
- **Whisper STT + OpenAI TTS** (and the `TranscriberProtocol` /
  `SpeakerProtocol` they sit behind)
- **Observability — Tracer, TraceWriter, cost.py, structured
  logging, correlation IDs**
- **Docker + HF Spaces** deployment pipeline
- All **locked schemas** at `schema_version="1.0.0"`

Every change proposed in the next two documents preserves the
above. We are upgrading the *experience and visualization layer
on top* of this architecture — not the architecture itself.
