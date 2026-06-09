# Clarion

**Configurable Multi-Agent Voice Automation Platform with Sentinel Trust Engine.**

Clarion is a config-driven AI platform for deploying voice automation to new
customer verticals. The demonstration vertical is healthcare scheduling — a
patient calls a specialty practice to book, reschedule, cancel, or ask a
routine question, and Clarion handles it against per-customer rules with a
trust engine that decides when to escalate to a human.

The architecture is built so a new vertical is onboarded by dropping in a
config + rules file, not by rewriting code. That's the Forward Deployed
Engineer story this project demonstrates.

> **Honesty note.** This is a prototype on synthetic, non-PHI data. Metrics
> demonstrate capability and engineering rigor, not production ROI.

## Status

Phase 11 complete — every harness scenario carries an `EscalationScore`
with five sub-signals (low_confidence, repeated_clarification,
rule_conflict, frustration, unsupported_request) plus an
"already_escalated" short-circuit that treats guardrail short-circuits
as the strongest predictor. `stats_from_run(scenarios, report)` folds
the 100-scenario set into an `EscalationStats` (precision, recall, F1,
accuracy, confusion matrix). 100% emergency recall on both customers;
overall precision above 0.5 on both. Phases 10 (Sentinel LLM-as-judge)
and 11 (escalation engine) plug into `HarnessResult` so Phase 12 can
fold them into the consolidated evaluation report.

## Build phases

| Phase | Scope | Status |
| ----- | ----- | ------ |
| 1     | Foundation (Poetry, Ruff, Pytest, Docker, CI) | ✅ complete |
| 2     | Multi-tenant config system                    | ✅ complete |
| 3     | Dual data pipeline (RAG + SQLite)             | ✅ complete |
| 4     | Schemas + mocked tools                        | ✅ complete |
| 5     | Text agent MVP (ReAct)                        | ✅ complete |
| 6     | Guardrails (emergency / clinical / PHI)       | ✅ complete |
| 7     | Observability (tokens, latency, cost, traces) | ✅ complete |
| 8     | FastAPI service                               | ✅ complete |
| 9     | Simulation harness                            | ✅ complete |
| 10    | Sentinel trust engine (LLM-as-judge)          | ✅ complete |
| 11    | Escalation engine (5-signal score + P/R)      | ✅ complete |
| 12    | Evaluation framework                          | pending |
| 13    | Streamlit dashboard                           | pending |
| 14    | Deployment (Cloud Run)                        | pending |
| 15    | Documentation                                 | pending |
| 16    | LangGraph refactor                            | pending |
| 17    | Voice layer                                   | pending |
| 18    | Emotion detection                             | pending |
| 19    | Production hardening                          | pending |
| 20    | v1.0.0 release                                | pending |

## Quick start

```bash
# Install Poetry if you don't have it
pip install --user poetry

# Install dependencies
poetry install --with dev

# Lint + type-check + test
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
poetry run pytest
```

## Boot from a customer config

Every deployment selects exactly one customer via the `CLARION_CUSTOMER`
environment variable. The selector matches a YAML stem in `configs/`.

```bash
# Default — ophthalmology
poetry run python -c "from clarion.config import load_customer; print(load_customer())"

# Switch to a different customer with no code change
CLARION_CUSTOMER=orthopedics poetry run python -c \
  "from clarion.config import load_customer; c=load_customer(); print(c.display_name, c.enabled_tools)"
```

Adding a third customer is one file: drop `configs/<name>.yaml`, set
`CLARION_CUSTOMER=<name>`. No agent-code change. That's the FDE story this
project demonstrates.

The shipped configs:

| Customer | Tools enabled | Languages | Notable divergence |
| -------- | ------------- | --------- | ------------------ |
| `ophthalmology` | all 5 | en, es | full tool surface |
| `orthopedics`   | 4 (no `cancel_appointment`) | en | cancellations always route to a human; stricter escalation thresholds |

## Dual data pipeline

Each customer has two data stores, both populated by one CLI:

```bash
# Build both pipelines for one customer (runs structured + unstructured)
poetry run python -m clarion.pipelines.ingest all ophthalmology
poetry run python -m clarion.pipelines.ingest all orthopedics

# Or just one half
poetry run python -m clarion.pipelines.ingest structured   ophthalmology
poetry run python -m clarion.pipelines.ingest unstructured orthopedics
```

**Structured pipeline (`clarion.pipelines.structured`)** — SQLite, one file
per customer at `data/<customer_id>/structured.sqlite`. Tables: `providers`,
`availability`, `appointments`, `eligibility`. The agent's tools (Phase 4)
call into `StructuredStore` and never see raw SQL. Seed data lives in
`data/seeds/<customer_id>.json`.

**Unstructured pipeline (`clarion.pipelines.unstructured` + `clarion.rag`)** —
markdown rules in `data/rules/<customer_id>/*.md` are chunked on H2/H3
boundaries (preserving heading trail as metadata), embedded with TF-IDF by
default (1-2 ngrams, sublinear TF, L2-normalized dense float32), and indexed
with `faiss.IndexFlatIP` so inner product equals cosine similarity. Setting
`OPENAI_API_KEY` swaps the default embedder to `text-embedding-3-small`
without any code change.

```python
from clarion.config import load_customer
from clarion.rag import load_customer_retriever
from pathlib import Path

cfg = load_customer("ophthalmology")
retriever = load_customer_retriever(cfg, data_dir=Path("data"))
for hit in retriever.retrieve("how long is a cataract consult?", k=3):
    print(round(hit.score, 3), hit.chunk.heading, "—", hit.chunk.source)
```

Acceptance verified by `tests/rag/test_retriever.py` and
`tests/pipelines/test_structured_store.py`:

- **RAG retrieves correct rules** — 10 parametrized queries (5 per customer)
  assert the right file appears in top-3, plus cross-customer isolation
  checks (workers-comp content never surfaces in ophthalmology; dilation
  content never surfaces in orthopedics).
- **Structured queries return valid records** — 16 tests over upsert,
  search, atomic booking with double-book protection, cancellation
  idempotence, and per-customer SQLite isolation.

## Repository layout

```
clarion/
├── README.md
├── LICENSE
├── pyproject.toml         # Poetry + ruff + mypy + pytest config
├── Dockerfile             # multi-stage, py3.11-slim, non-root
├── .pre-commit-config.yaml
├── .github/workflows/ci.yml
│
├── docs/                  # discovery doc, architecture diagram
├── configs/               # per-customer YAML (ophthalmology, orthopedics)
├── data/
│   ├── rules/             # unstructured rules corpora (markdown, per-customer)
│   ├── seeds/             # structured-pipeline seeds (JSON, per-customer)
│   ├── personas/          # synthetic patient personas (Phase 9)
│   └── <customer_id>/     # generated: structured.sqlite + rules.faiss (gitignored)
│
├── clarion/               # importable package
│   ├── config/            # Pydantic Settings, customer loader
│   ├── pipelines/         # unstructured + structured ingest
│   ├── rag/               # FAISS retriever (+ TF-IDF fallback)
│   ├── agents/            # router → specialist → supervisor
│   ├── tools/             # search_slots, book_appointment, …
│   ├── schemas/           # Pydantic models
│   ├── sentinel/          # judge, escalation scorer, guardrails
│   ├── observability/     # token / latency / cost meter + trace spans
│   ├── simulator/         # synthetic patient generator
│   ├── evaluation/        # harness runner + metrics
│   └── voice/             # STT / TTS / emotion adapter
│
├── api/                   # FastAPI app (Phase 8)
├── dashboard/             # Streamlit dashboard (Phase 13)
└── tests/
```

## Tools

Every tool follows the same shape — Pydantic input, Pydantic output, never
raises to the agent:

```python
from clarion.config import load_customer
from clarion.pipelines.structured import StructuredStore
from clarion.tools import ToolContext, available_tools, get_tool
from clarion.schemas import SearchSlotsInput
from datetime import date
from pathlib import Path

cfg = load_customer("ophthalmology")
ctx = ToolContext(
    customer=cfg,
    structured=StructuredStore.for_customer(cfg.customer_id, Path("data")),
)

# Discover what the agent can call for this customer
for tool in available_tools(cfg):
    print(tool.name)

# Use one
tool = get_tool("search_slots", cfg)
out = tool.run(
    SearchSlotsInput(appointment_type="Cataract Pre-Op Consult", on_or_after=date.today()),
    ctx,
)
if out.ok:
    for slot in out.slots:
        print(slot.slot_id, slot.slot_date, slot.start_time)
else:
    print("escalate:", out.error)
```

The five shipped tools:

| Tool | What it does | Failure modes returned as `ok=False` |
| ---- | ------------ | ------------------------------------ |
| `search_slots` | List open slots filtered by type / provider / start date | DB error |
| `book_appointment` | Atomically reserve one slot for one patient | slot unavailable (gone or double-book) |
| `cancel_appointment` | Cancel by id, free the slot | (idempotent — unknown id returns `ok=True, cancelled=False`) |
| `check_eligibility` | Look up the patient's payer record | (unknown patient returns `ok=True, on_file=False`) |
| `create_pms_task` | File a follow-up task for the front desk | DB error |

**Per-customer enforcement.** The registry honors each customer's
`enabled_tools` list. `configs/orthopedics.yaml` deliberately omits
`cancel_appointment` — when the agent calls `get_tool("cancel_appointment", cfg)`
for orthopedics, it raises `ToolNotEnabledError` listing what *is* enabled,
so the LLM never sees a tool the customer disabled.

**Retries.** SQLite call sites are wrapped in `run_with_retry` (one retry
on `OperationalError`, ~50 ms backoff) to absorb transient contention. The
helper is unbounded so individual tools can apply it however they need.

## Text agent

The agent is a ReAct loop wrapped around an `LLMClient` Protocol:

```python
from pathlib import Path
from clarion.agents import Agent, OpenAIClient  # OpenAIClient lives in clarion.agents.openai_client
from clarion.config import load_customer

cfg = load_customer("ophthalmology")
agent = Agent.build(
    customer=cfg,
    llm=OpenAIClient(),  # requires OPENAI_API_KEY
    data_dir=Path("data"),
)
print(agent.chat("Hi, I'd like to book a cataract pre-op consult after June 1."))
print(agent.chat("My patient id is pat_001."))
print(agent.chat("Yes, 9 AM works."))
```

Quick local REPL:

```bash
OPENAI_API_KEY=sk-... poetry run python -m clarion.agents.cli ophthalmology
# or
OPENAI_API_KEY=sk-... poetry run python -m clarion.agents.cli orthopedics
```

The ingest CLI must have been run first
(`python -m clarion.pipelines.ingest all <customer>`) so the FAISS index
and SQLite store exist on disk.

### How a turn works

1. `agent.chat(user_message)` appends the user turn to a rolling
   transcript.
2. The system prompt is rebuilt for this turn — persona from
   `CustomerConfig.agent_persona`, a hardcoded operating contract
   (no clinical advice, escalate emergencies, call tools rather than
   guess), the customer's escalation thresholds, and top-k chunks
   retrieved against the *current* user message (so RAG focuses on the
   topic the patient just raised, not the whole call history).
3. `react_loop` advertises only the customer's enabled tools, runs
   `llm.complete(...)`, validates any tool calls against each tool's
   Pydantic input model, dispatches them, and feeds the JSON results
   back to the LLM as `role="tool"` messages.
4. The loop stops when the LLM emits free text (the patient's reply) or
   after `max_steps` (default 8) — the cap returns a polite "let me
   have a teammate call you back" line and flags the result for the
   Sentinel.

### Capabilities

| Capability | How it's covered |
| ---------- | ---------------- |
| Booking | `search_slots` + `book_appointment` |
| Cancellation | `cancel_appointment` (when customer enables it) |
| Rescheduling | Composed from `search_slots` + `book_appointment` + `create_pms_task` to cancel the old slot when the cancel tool isn't enabled |
| Eligibility checking | `check_eligibility` (returns `on_file=False` for unknown patients without erroring) |
| FAQ retrieval | Per-turn RAG injects top-k rule chunks into the system prompt — the LLM cites the source markdown file |
| Function calling | Pydantic input models → JSON Schema via `clarion.agents.openai_schema.tool_to_spec` |
| Conversation memory | `Agent.transcript: list[Message]` — every user / assistant / tool turn except the system message (rebuilt each turn) |

### Tested end-to-end on both customers

| Test file | Scenario |
| --------- | -------- |
| `tests/agents/test_e2e_ophthalmology.py` | Patient books cataract pre-op — agent runs check_eligibility → search_slots → book_appointment → create_pms_task → confirms. Slot is gone from the store; transport task is filed. |
| `tests/agents/test_e2e_ophthalmology.py` | "Do you accept Kaiser?" — pure FAQ, zero tool calls; the LLM reads the answer from the rules block that RAG injected into the system prompt. |
| `tests/agents/test_e2e_orthopedics.py` | Patient asks to cancel — `cancel_appointment` is disabled for this customer, the registry returns `not enabled`, the next LLM turn files a PMS task per the practice's "all cancellations go to a human" rule. |
| `tests/agents/test_e2e_orthopedics.py` | Inspection check — the tools advertised to the LLM are exactly the four orthopedics permits (no `cancel_appointment` ever leaks into the spec list). |
| `tests/agents/test_e2e_orthopedics.py` | Reschedule emerges from `search_slots` + `book_appointment` + `create_pms_task` to cancel the old slot — no per-tenant code path. |

## Guardrails

Guardrails run **before** the ReAct loop sees a user message. If any
fire, the LLM is bypassed — short-circuiting with a canned, customer-
neutral reply.

```python
from pathlib import Path
from clarion.agents import Agent
from clarion.config import load_customer
from clarion.sentinel import AuditLog
# ... build store, retriever, llm as before ...

agent = Agent.from_customer(
    customer=load_customer("ophthalmology"),
    llm=llm,
    structured=store,
    retriever=retriever,
)
agent.audit = AuditLog.for_customer("ophthalmology", data_dir=Path("data"))
# guardrails_enabled defaults to True; opt-out with agent.guardrails_enabled = False

print(agent.chat("I suddenly lost my sight in my right eye!"))
# → "This sounds like an emergency. Please call 911 or go to the nearest
#    emergency department right now. I've also flagged this for our care
#    team so they can follow up."
# (LLM never consulted; urgent PMS task filed; audit row with guardrail="emergency")
```

### What's covered

| Guardrail | Trigger examples | Agent response | Side effects |
| --------- | ---------------- | -------------- | ------------ |
| Emergency | "I can't see", "compound fracture", "having a stroke", "I called 911", "lost control of my bladder", "this is an emergency" | Canned 911 / ED advisory | Urgent PMS task filed (`priority="urgent"`); audit `guardrail="emergency"` with `escalation_task_id` |
| Clinical advice | "should I take/stop/double/skip X", "is it safe to Y", "can I take A with B", "what dose", "refill my prescription/medication/drops" | Canned "I'll have a clinician call back" | Audit `guardrail="clinical_advice"`; no PMS task |
| PHI redaction | Applied to **all** audit log records (including LLM-handled turns) | — | User message + agent reply + tool-call arguments all sanitized: `<PHONE>`, `<SSN>`, `<EMAIL>`, `<MEMBER_ID>`, `<PATIENT_ID>` |
| Audit log | Every chat turn | — | Append-only JSONL at `<data_dir>/<customer_id>/audit.jsonl` with timestamp, conversation_id, customer_id, redacted messages, redaction counts, guardrail flag, tool calls, step count |

### Trigger curation

Patterns live in [`clarion/sentinel/guardrails.py`](clarion/sentinel/guardrails.py) and are derived from each customer's `06_emergencies_and_escalation.md`. They are deliberately pattern-based (not LLM-based) so they're auditable and zero-latency. An LLM-as-judge layer (Phase 10) can vet ambiguous cases — the regex layer remains the cheap, deterministic floor.

### Negative coverage

The same test files also include 21 negative examples — "Should I bring sunglasses?", "Is it ok to reschedule?", "refill my coffee", "I broke my glasses", "mild knee pain" — all of which must **not** trip a guardrail. These prove the patterns aren't over-eager.

## Observability

Every `Agent.chat` call emits one `Trace` with hierarchical `Span`s.
Attach a `TraceWriter` to persist them as JSONL:

```python
from pathlib import Path
from clarion.agents import Agent
from clarion.config import load_customer
from clarion.observability import TraceWriter
from clarion.sentinel import AuditLog

cfg = load_customer("ophthalmology")
agent = Agent.build(customer=cfg, llm=llm, data_dir=Path("data"))
agent.audit  = AuditLog.for_customer(cfg.customer_id, Path("data"))
agent.traces = TraceWriter.for_customer(cfg.customer_id, Path("data"))
agent.chat("Hi, I'd like a cataract pre-op consult after June 1.")
# Writes one line to data/ophthalmology/traces.jsonl with the full span tree.
```

### Span hierarchy

```
agent.chat                    user_chars, reply_chars
├── guardrails.check          fired: bool
├── retrieval                 k, hit_count, top_score, top_source
├── react.step                step_index
│   ├── llm.complete          model, input_tokens, output_tokens,
│   │                         cost_usd, advertised_tools, tool_calls_count
│   ├── tool.search_slots     tool, ok, arguments_keys
│   └── tool.book_appointment ...
├── react.step
│   ├── llm.complete
│   └── tool.create_pms_task
└── react.step
    └── llm.complete          (final text reply, no tools)
```

### What's tracked

| Spec line | How it's covered |
| --------- | ---------------- |
| Token usage | `LLMUsage(model, input_tokens, output_tokens)` populated from `OpenAIClient.complete`; flows into `llm.complete.attributes` |
| Latency | Every span carries `duration_ms` from `time.perf_counter` (sub-ms accuracy) |
| Cost | `cost_usd` per-model pricing table in `clarion/observability/cost.py`; unknown models report `0.0` so the gap is visible, not silently absorbed |
| Tool usage | One `tool.<name>` span per call with `ok`, `arguments_keys`, optional `error` (truncated to 200 chars; arg values stay out of traces — that's the audit log's job) |
| Retrieval quality | `retrieval` span with `hit_count`, `top_score`, `top_source` (citation file) |
| JSON traces | `TraceWriter` appends one line per turn to `<data_dir>/<customer_id>/traces.jsonl` |
| Span logging | `Tracer` is a stack-based context manager; parent/child links via `parent_id` |
| Trace IDs | `trace_<12hex>`; auto-generated per chat; surfaced in the audit row's `extra.trace_id` so audit ↔ trace pivots are one-click |

### Pricing table

Currently shipped (`clarion/observability/cost.py`):

| Model | Input ($/1k tokens) | Output ($/1k tokens) |
| ----- | ------------------- | -------------------- |
| `gpt-4o-mini` | 0.00015 | 0.0006 |
| `gpt-4o` | 0.0025 | 0.01 |
| `gpt-4-turbo` | 0.01 | 0.03 |

Add new models in one place; every `llm.complete` span picks up the new pricing automatically.

## HTTP API

Run the service:

```bash
OPENAI_API_KEY=sk-... poetry run python -m api.app --port 8080
# Or directly:
OPENAI_API_KEY=sk-... poetry run uvicorn api.app:app --host 0.0.0.0 --port 8080
```

Once it's up:

- **Swagger UI** — http://localhost:8080/docs
- **ReDoc** — http://localhost:8080/redoc
- **OpenAPI spec** — http://localhost:8080/openapi.json

### Endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| `GET`  | `/health` | Liveness probe; returns `status`, `version`, `customers_loaded` |
| `POST` | `/chat` | Single user turn for one conversation. Allocates a `conversation_id` if the client omits it; echo it back to continue. |
| `POST` | `/evaluate` | Run a scripted scenario (1–20 messages) and return per-turn replies plus aggregate metrics (tokens, cost, latency, tool usage). |

### Example: chat (curl)

```bash
# First turn — no conversation_id; the server allocates one
curl -s http://localhost:8080/chat \
  -H 'content-type: application/json' \
  -d '{"customer_id":"ophthalmology","message":"I want a cataract pre-op consult after June 1"}' \
  | jq

# Next turn — pass back the conversation_id from the previous response
curl -s http://localhost:8080/chat \
  -H 'content-type: application/json' \
  -d '{"customer_id":"ophthalmology","conversation_id":"conv_abc123","message":"9 AM works"}' \
  | jq
```

Response shape:

```json
{
  "customer_id": "ophthalmology",
  "conversation_id": "conv_abc123def456",
  "trace_id": "trace_def456abc789",
  "reply": "You're all set — Cataract Pre-Op Consult on June 15 at 9 AM with Dr. Smith."
}
```

### Example: evaluate

```bash
curl -s http://localhost:8080/evaluate \
  -H 'content-type: application/json' \
  -d '{
        "customer_id": "ophthalmology",
        "scenario_id": "smoke_book_cataract",
        "messages": [
          "hi",
          "I want a cataract pre-op consult after June 1"
        ]
      }' \
  | jq
```

Returns `transcript`, `trace_ids`, and a `metrics` block with `turns`,
`total_steps`, `total_input_tokens`, `total_output_tokens`,
`total_cost_usd`, `total_latency_ms`, `tools_used`.

### State + multi-tenancy

- **Per-customer** (`CustomerConfig`, `StructuredStore`, `Retriever`, `AuditLog`, `TraceWriter`) is lazily loaded on first request and cached for the lifetime of the process.
- **Per-conversation** Agents (one per `(customer_id, conversation_id)`) live in an LRU bounded by 256 sessions. Long-tail clients can't OOM the process.
- Each `/chat` request advances the same transcript when the client echoes the `conversation_id` back. Each `/evaluate` always allocates a fresh `conversation_id` so scenario runs are isolated.
- Audit + trace writers per customer remain shared, so all conversations for one tenant land in the same files. The Phase 13 dashboard reads them directly.

### Error contract

| Status | `code` | Meaning |
| ------ | ------ | ------- |
| 404    | `customer_not_found` | No YAML matching `customer_id` |
| 400    | `customer_config_invalid` | YAML exists but failed schema validation |
| 422    | (Pydantic default) | Request body fails validation (e.g. uppercase `customer_id`, empty `message`) |

### Tests

`tests/api/` runs entirely via `fastapi.testclient.TestClient` — **no real
LLM calls in CI**. The conftest fixture wires a `FakeLLM` factory into
the `SessionManager` so every test scripts the LLM's responses
deterministically. 15 tests cover health, chat (happy path, tool-calling
turn, transcript continuity, multi-tenant divergence, guardrail
short-circuit, error mapping), and evaluate (scenario aggregation,
mid-scenario guardrail handling, validation errors).

## Simulation harness

`clarion.simulator` produces 100 synthetic patient scenarios per
customer and runs them through the real Agent for end-to-end grading.

```bash
# Regenerate the persona JSONs (idempotent — same seed each time)
poetry run python -m clarion.simulator.cli generate ophthalmology
poetry run python -m clarion.simulator.cli generate orthopedics
poetry run python -m clarion.simulator.cli generate all

# Run scripted-mode harness (no API key, deterministic, ~5s/customer)
poetry run python -m clarion.simulator.cli run all --report-out reports/

# Run live-mode against real OpenAI (requires OPENAI_API_KEY)
poetry run python -m clarion.simulator.cli run ophthalmology --mode live
```

Sample output:

```
ophthalmology: 100/100 passed (100.0%); failed=0
  difficulty=ambiguous       15/ 15 passed
  difficulty=clear           69/ 69 passed
  difficulty=emergency       10/ 10 passed
  difficulty=rule_violating   6/  6 passed
  intent=book                41/ 41 passed
  intent=cancel               8/  8 passed
  intent=clinical_advice     10/ 10 passed
  intent=eligibility          8/  8 passed
  intent=emergency           10/ 10 passed
  intent=faq                 10/ 10 passed
  intent=reschedule          13/ 13 passed
```

### Scenario shape

Each scenario in `data/personas/<customer>.json` carries:

```json
{
  "scenario_id": "ophthalmology_clear_book_001",
  "customer_id": "ophthalmology",
  "difficulty": "clear",
  "intent": "book",
  "messages": ["Hi, this is Jane Smith, patient id pat_001..."],
  "ground_truth": {
    "expected_outcome": "booked",
    "should_escalate": false,
    "expected_tools": ["search_slots", "book_appointment"],
    "expected_appointment_type": "Cataract Pre-Op Consult",
    "notes": "Clear booking request with patient id provided."
  },
  "llm_script": [
    { "tool_calls": [{"name": "search_slots", "arguments": {...}}] },
    { "tool_calls": [{"name": "book_appointment", "arguments": {...}}] },
    { "content": "You're booked for Cataract Pre-Op Consult on June 15." }
  ]
}
```

The `llm_script` lets scripted mode replay deterministic agent behavior
without an API key. Live mode ignores it.

### Distribution across the 100 scenarios

| Difficulty / Intent | Count |
| ------------------- | ----- |
| clear / book        | 25 |
| clear / cancel      | 8 |
| clear / reschedule  | 8 |
| clear / eligibility | 8 |
| clear / faq         | 10 |
| clear / clinical_advice | 10 |
| ambiguous / book    | 10 |
| ambiguous / reschedule | 5 |
| rule_violating / book | 6 |
| emergency / emergency | 10 |

Total = 100. Per-tenant divergence is automatic — orthopedics' cancel
+ reschedule scenarios route to `create_pms_task` instead of
`cancel_appointment` because that's what its YAML says.

### Acceptance

Phase 9 spec: *"Harness can automatically run evaluations."* Met by:

- 274 tests in CI include `tests/simulator/test_harness.py` which runs
  all 200 scenarios across both customers and asserts 100% pass rate.
- `python -m clarion.simulator.cli run all` exits 0 on green, non-zero
  otherwise — gateable from CI.
- Each scenario carries enough ground truth that Phase 10's LLM-judge
  and Phase 12's metric suite can score richer dimensions on the same
  data without regenerating.

## Sentinel LLM-as-judge

The Sentinel trust engine has two layers stacked on each other:

1. **Pattern guardrails** (Phase 6) — pre-LLM regex checks for emergencies
   and clinical-advice requests. Fast, auditable, deterministic.
2. **LLM-as-judge** (Phase 10) — post-LLM grading that catches what
   regex can't: wrong-but-plausible bookings, hallucinated providers,
   subtle policy drift.

```python
from clarion.agents.openai_client import OpenAIClient
from clarion.schemas import JudgeRequest
from clarion.sentinel import Judge

judge = Judge(llm=OpenAIClient())   # any LLMClient works

verdict = judge.judge(
    JudgeRequest(
        customer_id="ophthalmology",
        user_message="Book me a cataract pre-op consult.",
        agent_reply="You're booked for a Routine Eye Exam on June 16.",
        tool_calls=[{"name": "book_appointment", "arguments": {...}, "ok": True}],
        rag_context=["Cataract Pre-Op Consult: 60 minutes.", ...],
        expected_appointment_type="Cataract Pre-Op Consult",
    )
)
verdict.booking_correct      # 0.1 — wrong appointment type
verdict.hallucination        # 0.0
verdict.policy_violations    # [PolicyViolation(kind="other", ...)]
verdict.passed               # False
verdict.rationale            # "Booking does not match user's intent."
```

### Dimensions

| Dimension | Range | What 0.0 means | What 1.0 means |
| --------- | ----- | -------------- | -------------- |
| `booking_correct` | 0.0–1.0 or `None` | wrong appointment type/patient/slot | matches user intent + practice rules |
| `hallucination` | 0.0–1.0 | every claim supported by rules / tool output | reply invents appointment types, providers, payers |
| `policy_violations` | list | empty | one or more flagged kinds (clinical_advice_given, emergency_not_escalated, invented_provider, invented_appointment_type, invented_payer_policy, phi_in_response, unsupported_claim, other) |
| `violation_severity` | 0.0–1.0 | trivial | safety-critical |
| `confidence` | 0.0–1.0 | judge couldn't parse / unsure | high confidence in the grade — Phase 11 escalation reads this |

### Self-reflection

`Judge.reflect(request)` is the same call wrapped in a second-person
voice ("you (Clarion)" instead of "Clarion") so the rationale reads as
a self-critique. Use it pre-publish; use `judge()` post-hoc. The
verdict shape is identical so callers can chain both without diverging.

### Wiring into the harness

```python
from clarion.sentinel import Judge
from clarion.simulator.harness import run_scripted

report = run_scripted(
    scenarios,
    customer_config=cfg,
    structured=store,
    retriever=retriever,
    judge=Judge(llm=OpenAIClient()),   # optional
)

for result in report.results:
    if result.judge_verdict and result.judge_verdict["policy_violations"]:
        print(result.scenario_id, "→", result.judge_verdict["rationale"])
```

When `judge` is omitted, every result's `judge_verdict` stays `None` —
backward-compatible with the Phase 9 harness contract.

### Defensive parsing

The LLM occasionally misbehaves: emits markdown-fenced JSON, drops a
required field, returns a string instead of an object, returns scores
outside [0, 1]. The judge handles all of these without raising:

* `` ```json ... ``` `` fences are stripped before parse.
* Malformed JSON → low-confidence verdict with `rationale="parse failure"`.
* Out-of-range scores are clamped to `[0, 1]`.
* Missing fields take sane defaults.

A `confidence=0.0` verdict is the operator's signal that something
upstream needs investigation.

### Acceptance

Spec: *"Injected errors are detected."* Proven by
`tests/sentinel/test_judge_acceptance.py`:

| Injected failure | Test name | Judge response |
| ---------------- | --------- | -------------- |
| Wrong appointment type (cataract requested, routine booked) | `test_judge_catches_wrong_appointment_type` | `booking_correct < 0.5` + policy violation |
| Hallucinated provider "Dr. Random" | `test_judge_catches_hallucinated_provider` | `hallucination >= 0.7` + `invented_provider` violation |
| Emergency phrase booked instead of escalated | `test_judge_catches_emergency_that_was_not_escalated` | `emergency_not_escalated` violation with `severity >= 0.8` |

Plus an inverse sanity test (`test_judge_passes_a_correct_booking`) so
a regression where the judge always fails is caught.

## Escalation engine

`clarion.sentinel.escalation` produces a 0–1 escalation score per
conversation turn, derived from five signals plus an "already escalated"
short-circuit. Each scenario in the harness gets one attached:

```python
from clarion.sentinel.escalation import (
    EscalationScorer,
    ConversationFacts,
    stats_from_run,
)
from clarion.simulator.harness import load_scenarios, run_scripted

scenarios = load_scenarios("data/personas/ophthalmology.json")
report = run_scripted(scenarios, ...)        # populates result.escalation
stats = stats_from_run(scenarios, report)    # precision / recall / F1
print(stats)
```

### Signals

| Signal | Source | Notes |
| ------ | ------ | ----- |
| `low_confidence` | `1 - judge.confidence` if Judge attached; else regex on "I'm not sure" patterns in the agent reply | When no judge is wired, falls back to text heuristic |
| `repeated_clarification` | count(agent replies ending in `?`) / `customer.escalation.max_clarifications` | Per-customer threshold from YAML |
| `rule_conflict` | Judge flagged any `unsupported_claim` or `invented_*` policy violation | Requires Judge; 0 otherwise |
| `frustration` | `detect_frustration_over_turns(user_messages).score` | Pattern-based: "I already told you", "let me speak to a manager", SHOUTING, `???`, etc |
| `unsupported_request` | 1.0 when `create_pms_task` was called AND scenario.expected_outcome != "task_created" | Orthopedics cancel scenarios don't trip this (task IS the expected outcome) |
| **`already_escalated`** (short-circuit) | A task was filed OR the reply contains "911" or "clinical" | When True, score=1.0; bypasses the weighted sum |

### Composite

```
score = clip(0, 1, normalize(sum(signal * weight)))
should_escalate = score >= decision_threshold   # default 0.5
```

Default weights (sum to 1.0): `low_confidence=0.30`,
`repeated_clarification=0.15`, `rule_conflict=0.20`, `frustration=0.20`,
`unsupported_request=0.15`. Tuned so any single severe signal crosses
0.5, or two mild ones together do.

### Precision / Recall acceptance

The Phase 11 acceptance test runs all 100 scenarios per customer through
the scripted harness and asserts:

- `stats.total == 100` and all four confusion-matrix cells sum to 100
- `recall == 1.0` on the 10 emergency-intent scenarios (no misses
  allowed — these are the highest-stakes category)
- `precision >= 0.5` overall (guards against a "predict True always"
  scorer degeneration)

Verified for both shipped customers. The `stats_from_run` helper takes a
`HarnessReport` + the original scenarios and returns an
`EscalationStats` ready to be folded into the Phase 12 evaluation report.

## Design principles

1. **Text core first, voice as a shell.** Agent input is a transcript, output
   is text — from day one. Voice wraps a working, evaluated core later.
2. **Config-driven multi-tenancy from day one.** No hardcoded clinic. Rules,
   tools, escalation thresholds, persona — all from per-customer YAML.
3. **Evaluation is a first-class citizen.** The simulated-patient harness +
   Sentinel are the differentiator, not an afterthought.
4. **Ship at the Week-2 line.** Multi-agent (LangGraph) and voice are
   Week-3 upgrades on a complete, deployed spine.

## License

MIT — see [LICENSE](LICENSE).
