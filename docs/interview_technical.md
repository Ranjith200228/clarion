# Clarion — Technical Interview Prep Guide

> Use this document to confidently answer deep technical questions from ML engineers, software architects, and technical leads. Every section maps to a likely interview angle.

---

## 0. Opening 30-Second Pitch (say this first, every time)

> "Clarion is a production-grade, multi-tenant, multi-agent AI platform I built for healthcare front-desk automation. The core problem it solves is that specialty medical practices — think ophthalmology or orthopedics — receive extremely high call volume for routine tasks like booking and eligibility checks, but also zero-tolerance-for-failure calls like a patient reporting sudden vision loss. I built a five-layer system: a LangGraph multi-agent backend with five specialist nodes, an independent Sentinel trust engine that grades every AI reply from outside, a FastAPI service, an 11-tab Gradio dashboard, and a locked evaluation harness with 100 synthetic scenarios per tenant. The system achieves 100% booking accuracy, 100% safety catch rate, and ships with 705 tests, mypy strict, and a live HuggingFace Space deployment."

---

## 1. Business Problem & Why It's Hard

### The Domain
Specialty medical practices (ophthalmology, orthopedics) run front desks that handle:
- Appointment booking, rescheduling, cancellations
- Insurance eligibility checks
- Policy questions (hours, prep instructions, accepted payers)
- Emergency triage (sudden vision loss, acute fracture pain)

### Why a Generic LLM Agent Fails

**Problem 1: Auditability.** When a patient claims "the AI told me the appointment was at 2pm" and the chart says 3pm, you need a full audit trail. Most LLM agent frameworks produce either no trace or an opaque one.

**Problem 2: Safety bounds.** An emergency message like "I can't see anything in my left eye" should never reach the LLM — the LLM might hallucinate a home remedy, equivocate, or simply not escalate fast enough. The guardrail must fire before the LLM is ever invoked.

**Problem 3: Multi-tenancy without code branches.** Ophthalmology allows cancellation via the bot; orthopedics routes all cancellations to a human PMS task (workers' comp complexity). Solving this with `if customer_id == "ophthalmology"` in the agent loop creates unmaintainable spaghetti. The solution must be config-driven.

---

## 2. Five-Layer Architecture — Detailed

```
Layer 5:  11-tab Gradio Dashboard         (read-only consumer of schemas)
Layer 4:  FastAPI Service + Eval Harness  (REST API + 100-scenario corpus)
Layer 3:  Sentinel Trust Engine           (guardrails + judge + escalation)
Layer 2:  LangGraph Multi-Agent Core     (router → specialist → supervisor)
Layer 1:  Foundation                      (YAML config + SQLite + FAISS + tools)
```

Each layer has a **one-way import dependency** on the layer below. The dashboard cannot import the evaluation pipeline — structurally enforced by the Python module graph. This means a new metric is always a backend change that automatically propagates to the dashboard via the typed schema contract. The UI can never diverge from the source of truth.

---

## 3. Every Tool — What, Where, Why

### 3.1 LangGraph (StateGraph)
- **What:** Graph-based multi-agent orchestration framework from LangChain.
- **Where:** `clarion/multiagent/` — the entire multi-agent backend.
- **Why over plain ReAct:** LangGraph gives you explicit, auditable graph edges. You can read the graph and understand the routing logic without executing it. In a monolithic ReAct loop, you can't statically reason about which tool can be called by which context. LangGraph also has a first-class supervisor pattern that controls loop termination.
- **Specifically how it's used:**
  - `StateGraph` with `MultiAgentState` (TypedDict) as shared state
  - Nodes: `router_node`, `booking_node`, `info_node`, `emergency_node`, `eligibility_node`, `cancel_node`, `supervisor_node`
  - Conditional edges based on `state["intent"]` (router output) and `state["supervisor_decision"]`
  - TypedDict chosen over Pydantic for state because LangGraph's reducers (`Annotated[list, operator.add]`) work on plain dicts; Pydantic copies on every mutation which is expensive per-hop.

### 3.2 FastAPI
- **What:** ASGI Python web framework with automatic OpenAPI generation.
- **Where:** `api/` — all REST endpoints.
- **Why:** Pydantic v2 model → FastAPI → OpenAPI spec is automatic. Every request/response is fully typed. Adding a new endpoint means writing a Pydantic model, not writing validation code.
- **Endpoints:**
  - `POST /chat` — text turn through the agent
  - `POST /voice/turn` — base64 audio in, base64 audio out
  - `POST /cost/extract-invoice` — multipart image/PDF → structured line items
  - `GET /health` — liveness probe for Docker + HF Spaces

### 3.3 Gradio (Blocks)
- **What:** Python-native UI framework for ML demos, backed by Gradio.
- **Where:** `gradio_app/` — 11-tab dashboard.
- **Why:** Gradio Blocks lets you build a rich multi-tab dashboard entirely in Python with no JavaScript. The design-token CSS layer (~1,300 lines) gives it a professional look while keeping the underlying code readable.
- **Architecture decision:** The dashboard is a **read-only consumer** of the schema package. It calls `data_sources.py` roll-up functions that return typed dataclasses. It never computes a metric. This means the dashboard can't silently redefine a metric — it's structurally impossible without changing the backend schema.

### 3.4 Pydantic v2
- **What:** Data validation library for Python with Rust-backed parsing.
- **Where:** Every cross-module boundary — tool inputs, API request/response, evaluation report, dashboard wire models.
- **Why v2 over v1:** v2 generates JSON Schema that's compatible with OpenAI's tool-calling API. `extra="forbid"` on every model means unexpected fields raise a validation error at the boundary rather than being silently ignored.
- **Key models:**
  - `BookAppointmentInput` — five regex-validated fields, `extra="forbid"`
  - `EvaluationReport` — locked at v1.0.0; schema regression-gated at CI
  - `VoiceTurnRequest / VoiceTurnResponse` — base64 audio + metadata
  - `MultiAgentState` — TypedDict (not Pydantic, for LangGraph reducer compatibility)

### 3.5 OpenAI gpt-4o-mini
- **What:** OpenAI's cost-efficient GPT-4-class model.
- **Where used:**
  1. **Chat agent** — the specialist nodes call gpt-4o-mini with their tool specs
  2. **Intent router** — classifies user message to one of 5 specialist queues using a tool-call structured output (`route` tool)
  3. **LLM-as-Judge (Sentinel)** — post-turn verdict on hallucination, policy compliance, booking correctness
  4. **Invoice OCR** — Vision mode, JSON-mode output, structured line-item extraction
- **Why gpt-4o-mini over gpt-4o:** Cost-to-capability ratio for these tasks. Front-desk routing and eligibility checks don't need flagship reasoning. The judge is also gpt-4o-mini because it runs outside the agent — its job is pattern-checking, not creative reasoning.

### 3.6 OpenAI whisper-1 (STT)
- **What:** OpenAI's speech-to-text model.
- **Where:** `clarion/modules/voice/` — `OpenAIWhisperTranscriber` implements `TranscriberProtocol`.
- **Protocol pattern:** The voice module defines `TranscriberProtocol` with one method: `transcribe(audio_bytes, metadata) -> str`. Production uses `OpenAIWhisperTranscriber`. Tests use `EchoTranscriber` (returns the bytes decoded as UTF-8, or a canned string). The agent never knows which one is mounted.

### 3.7 OpenAI tts-1 (TTS)
- **What:** OpenAI's text-to-speech model.
- **Where:** `clarion/modules/voice/` — `OpenAITtsSpeaker` implements `SpeakerProtocol`.
- **Same protocol pattern:** `speak(text, metadata) -> bytes`. Tests use `SineWaveSpeaker` (generates a pure tone — valid audio bytes without an API key).

### 3.8 text-embedding-3-small
- **What:** OpenAI's embedding model, 1536 dimensions.
- **Where:** `clarion/pipelines/retriever.py` — FAISS-backed RAG system.
- **How RAG works:**
  1. At build time, rule Markdown files (e.g. `data/rules/ophthalmology/01_appointment_types.md`) are chunked and embedded → stored in a FAISS `IndexFlatIP` (inner product / cosine similarity).
  2. At runtime, user query → embed → FAISS search → top-k hits → prepend to specialist system prompt.
  3. The FAISS index is pre-baked in the Docker builder stage so the container starts instantly.

### 3.9 FAISS (Facebook AI Similarity Search)
- **What:** Facebook's library for fast approximate nearest-neighbor search.
- **Where:** `clarion/pipelines/` — one `IndexFlatIP` per tenant, serialized to `rules.faiss` + `rules_meta.json`.
- **Why FAISS over a vector DB like Pinecone:** Zero network latency, zero ops, zero cost. At the scale of 2 tenants with ~100 rule chunks each, a local index beats a hosted DB on every metric. The `.dockerignore` excludes the pre-built index files (host-platform binaries) and rebuilds them in the Docker builder stage.

### 3.10 SQLite
- **What:** Embedded relational database.
- **Where:** `clarion/pipelines/structured_store.py` — one `.sqlite` file per tenant.
- **Tables:** `appointments`, `patients`, `sessions`, `pms_tasks`
- **Why SQLite over Postgres:** Zero-ops for a single-container deployment. Deterministic for the eval harness (no external process). The Dockerfile rebuilds the SQLite store in the builder stage (excluded from Docker build context via `.dockerignore` to avoid shipping host-platform journal files).

### 3.11 XGBoost
- **What:** Gradient boosted decision tree library.
- **Where:** `clarion/modules/no_show_prediction/` — no-show risk predictor.
- **How it works:**
  1. Synthetic dataset generator produces ~10,000 patient records with features: `days_until_appointment`, `appointment_type`, `insurance_type`, `prior_no_shows`, `distance_km`, etc.
  2. XGBoost classifier trained on this data; serialized to `.joblib`.
  3. At inference: patient features → predictor → risk score (0–1) → risk band (low/medium/high).
  4. Top-decile lift metric (how much better than random at identifying the top 10% most likely no-shows) is reported in `EvaluationReport`.
- **Why XGBoost over logistic regression:** Handles non-linear feature interactions (e.g. "Monday morning + long commute" is a strong joint predictor) without feature engineering.

### 3.12 scikit-learn TF-IDF + Logistic Regression (BookingFastPath)
- **What:** Classic NLP pipeline for text classification.
- **Where:** `clarion/multiagent/booking_fastpath.py`
- **How it works:**
  1. 480 synthetic utterances for 5 booking intents (search/book/reschedule/cancel/check_eligibility) + fallback.
  2. TF-IDF vectorizer (max 3000 features, n-gram range 1–2) + LogisticRegression with L2 regularization.
  3. Trains in < 1 second in-process (no external training job).
  4. At inference: user turn → TF-IDF → LogReg → confidence score. If confidence ≥ 0.65, inject one-line hint into BookingSpecialist system prompt.
- **Why not fine-tune gpt-4o-mini:** Fine-tuning needs OpenAI's training API, labelled data pipeline, versioning, and costs money per call. This ships in-repo, owns its own failure modes, runs in 3 ms, and achieves ≥ 90% held-out accuracy on booking intent.
- **The key insight:** You're not bypassing the LLM. You're priming it. The LLM still owns the tool call and composes the reply. The hint just saves 40–80 tokens of "figure out what the user wants" reasoning per turn.

### 3.13 Docker (multi-stage)
- **What:** Container build system.
- **Where:** `Dockerfile` — three stages: `builder`, `test`, `runtime`.
- **Stage breakdown:**
  - `builder`: Python 3.11-slim, installs Poetry + all runtime deps, copies source, patches Gradio client, runs `build_indices.sh` (pre-bakes FAISS + SQLite), installs package root.
  - `test`: Extends builder, adds dev deps, runs `pytest -q`. `docker build --target test` is the CI acceptance gate — if any test fails, the build fails.
  - `runtime`: Python 3.11-slim, copies pre-built venv + source from builder, runs as non-root user (UID 1000 for HF Spaces compatibility), exposes 8000 and 7860, `ENTRYPOINT ["/usr/bin/tini", "--"]` for clean signal propagation.
- **Why pre-bake FAISS in builder:** Without this, the first `/chat` call would block on building the index. With it, the container is ready in < 2 seconds.

### 3.14 HuggingFace Spaces
- **What:** Free hosting for ML demo applications (Docker SDK).
- **Where:** `hf-space-deploy` git branch → pushed as `main` to the `space` remote.
- **How deployment works:** Two separate git remotes — `origin` (GitHub) and `space` (HF Spaces). The `hf-space-deploy` branch is an orphan with no binary history (avoids HF binary file rejection). Changes to `main` are ported with `git checkout main -- <files>`.
- **Key challenge solved:** `gradio_client/utils.py` crashes on Pydantic v2's `additionalProperties: True` JSON schema for `Dict[str, Any]`. Fixed via `scripts/patch_gradio_client.py` — a monkey-patch applied in the Docker builder stage that adds an `isinstance(schema, bool): return "Any"` guard to two functions in the gradio client.

### 3.15 GitHub Actions
- **What:** CI/CD platform.
- **Where:** `.github/workflows/` — matrix on Python 3.11 and 3.12.
- **What runs:** pytest (all 705 tests), ruff check, mypy --strict on `clarion` and `api` packages, schema regression test for `EvaluationReport v1.0.0`.

### 3.16 pytest (705 tests)
- **What:** Python testing framework.
- **Test categories:**
  - Unit tests: each component in isolation (FakeLLM stubs out OpenAI)
  - Integration tests: end-to-end booking flows per tenant
  - Schema regression: `EvaluationReport` v1.0.0 lock — additive fields pass, renaming/retyping fails
  - Load test (opt-in, `pytest -m loadtest`): 200 concurrent sessions with FakeLLM, validates p95 < 500 ms in-process
  - BookingFastPath classifier: 29 tests covering accuracy, per-intent correctness, noise robustness, specialist prompt integration

### 3.17 mypy --strict
- **What:** Static type checker for Python.
- **Where:** Runs on `clarion/` and `api/` packages.
- **Why strict:** `--strict` enables `disallow_untyped_defs`, `warn_return_any`, `no_implicit_optional`, etc. It forces every function to be fully typed, which means the type system catches cross-module contract violations at lint time, not at runtime.

### 3.18 ruff
- **What:** Extremely fast Python linter (replaces flake8, isort, pyupgrade).
- **Where:** Runs on all source packages in CI.

### 3.19 Poetry
- **What:** Python dependency and packaging manager.
- **Why over pip-tools:** Lockfile-based reproducibility, dev/ui/root dependency groups, first-class virtual environment management.

### 3.20 tini
- **What:** Minimal init process for containers.
- **Where:** Docker `ENTRYPOINT ["/usr/bin/tini", "--"]`
- **Why:** Docker containers run as PID 1 by default, which means they don't have a proper init process to reap zombie child processes or forward signals. When HF Spaces sends `SIGTERM` to shut down the container, tini forwards it to the Python process cleanly.

---

## 4. Multi-Agent Architecture — Deep Dive

### 4.1 The Five Specialists

| Specialist | Tools available | Notes |
|---|---|---|
| `booking` | `search_slots`, `book_appointment`, `reschedule_appointment` | BookingFastPath primes this node before LLM call |
| `eligibility` | `check_eligibility` | Insurance/payer queries |
| `info` | `rag_search` (FAISS retrieval) | Grounded on rules Markdown; cannot book |
| `cancel` | `cancel_appointment`, `create_pms_task` | Orthopedics omits `cancel_appointment` — all cancels go to PMS task |
| `emergency` | none | Deterministic short-circuit. No LLM call. Emits canned reply + PMS urgent task. |

### 4.2 The Routing Graph

```
User message
    │
    ▼
[Guardrails] ──fires──► Canned reply (LLM never called)
    │
    │ safe
    ▼
[Router node] ──► state["intent"] = "booking" | "eligibility" | ...
    │
    ▼
[Specialist node] ──► state["specialist_reply"]
    │
    ▼
[Supervisor node] ──► decision: finish | route | escalate
    │               route loops back to Router
    │               escalate flags for human handoff
    ▼
[Sentinel judge] ──► JudgeVerdict
    │
    ▼
[Escalation scorer] ──► EscalationScore (0–1 fusing 5 signals)
    │
    ▼
Final response to user
```

### 4.3 State Schema Design Decision
State is `MultiAgentState(TypedDict, total=False)` not a Pydantic model. Reason: LangGraph's reducer convention uses `Annotated[list, operator.add]` for append-only fields (the conversation transcript), which works on plain dicts. Pydantic models copy on every mutation — in a 6-node graph that's 6 copies per user turn. TypedDicts are flat and cheap.

### 4.4 Per-Specialist Tool Scoping
Each specialist receives only the tools defined in `configs/<tenant>.yaml → enabled_tools`. The tool list is filtered at specialist instantiation time:
```python
specialist_tools = [t for t in all_tools if t.name in config.enabled_tools]
```
The `info` specialist never sees `book_appointment`. The `emergency` specialist sees zero tools. This shrinks the attack surface for prompt injection: even if a user manages to trick the info specialist, it has no booking tool to misuse.

---

## 5. Sentinel Trust Engine — Deep Dive

### 5.1 Pre-LLM Guardrails (`clarion/sentinel/guardrails.py`)
- Runs on the raw user message **before** the LangGraph graph is invoked.
- Three kinds: `emergency`, `clinical_advice`, `safe`.
- Emergency patterns use word-boundary regex compiled at import time (e.g., `r"\bsudden(?:ly)?\s+(?:lost|losing)\s+(?:my\s+)?(?:sight|vision)\b"`).
- If guardrail fires: return canned response appropriate to tenant, file a `create_pms_task` with urgency=high. The LLM is **never invoked**.
- Failure mode is explicit: biases toward false alarms (escalating a non-emergency) over silent failures (letting an emergency through).

### 5.2 LLM-as-Judge (`clarion/sentinel/judge.py`)
- Runs after the specialist responds.
- Same gpt-4o-mini, different system prompt: "You are an independent quality auditor..."
- Outputs structured `JudgeVerdict`:
  - `booking_correct: bool`
  - `hallucination_detected: bool`
  - `policy_violation: Optional[str]`
  - `confidence: float` (0–1)
- Defensive JSON parsing: strips code fences, handles malformed output by defaulting to low-confidence rather than crashing.
- **Key design insight:** The judge runs outside the agent's context. It doesn't share the agent's reasoning chain, so it can't rationalize the same mistake. Two real bugs found in development: (1) agent hallucinated a specialty ophthalmology doesn't offer ("contact lens fitting" — the tenant is surgical-only), (2) agent confirmed 2:00 PM booking when the slot was 2:30 PM. Both looked correct from inside the agent; judge caught both.

### 5.3 Escalation Scorer (`clarion/sentinel/escalation.py`)
- Five weighted signals, each in [0, 1]:
  1. `low_confidence` — inverted from judge's `confidence` field (confidence 0.3 → low_confidence 0.7)
  2. `repeated_clarification` — agent asked 2+ clarifying questions (saturates against `max_clarifications` in tenant YAML)
  3. `rule_conflict` — judge flagged `unsupported_claim` or invented policy
  4. `frustration` — output of `detect_frustration_over_turns` (pattern-matched across transcript)
  5. `unsupported_request` — agent called `create_pms_task` because it couldn't handle inline
- Composite: `sum(signal * weight)`, clamped [0, 1].
- Each fired signal contributes a reason string to `EscalationScore.reasons` → renders in Sentinel Operations tab without re-computation.
- Thresholds are in tenant YAML — ophthalmology (where "my eye hurts" is urgent) has lower thresholds than orthopedics.

---

## 6. RAG System — How It Works End-to-End

1. **Indexing (build time):**
   - Rule Markdown files in `data/rules/<tenant>/` (01–06 covering appointment types, intake, prep, insurance, cancellation, emergencies)
   - Each file is chunked (~500 tokens with overlap)
   - Each chunk → `text-embedding-3-small` → 1536-dim float vector
   - Vectors stored in FAISS `IndexFlatIP` (inner product = cosine similarity on normalized vectors)
   - Index serialized to `rules.faiss` + metadata to `rules_meta.json`

2. **Inference (runtime):**
   - User message → `text-embedding-3-small` → query vector
   - FAISS `search(query, k=3)` → top-3 rule chunks
   - Chunks prepended to specialist system prompt as "RELEVANT RULES: ..."
   - The `info` specialist is the primary consumer; booking specialist also uses RAG for policy questions embedded in booking flows

3. **Reason for FAISS over a simpler BM25:** The rules documents mix clinical terminology with lay language (a patient says "blurry" not "reduced visual acuity"). Semantic similarity handles this better than keyword matching.

---

## 7. Voice Pipeline — End-to-End

```
Client ──base64 WAV──► POST /voice/turn
                           │
                           ├─ decode base64 + validate n_bytes
                           │
                           ├─ SessionManager.get_or_create_session(customer_id, session_id)
                           │
                           ├─ VoiceOrchestrator.turn(agent, audio_bytes)
                           │      │
                           │      ├─ OpenAIWhisperTranscriber.transcribe(audio_bytes) → text
                           │      │
                           │      ├─ agent.chat(text) → reply_text
                           │      │   (full LangGraph multi-agent pipeline)
                           │      │
                           │      └─ OpenAITtsSpeaker.speak(reply_text) → audio_bytes
                           │
                           └─ return VoiceTurnResponse(audio_b64, transcript, reply, latencies)
```

**Session continuity:** `session_id` is passed by the client. The `SessionManager` maps `(customer_id, session_id)` → `(session, agent)`. A voice conversation and a chat conversation with the same `session_id` share the same rolling transcript — you can switch modalities mid-conversation.

**Error handling:** OpenAI auth/quota errors (`AuthenticationError`, `PermissionDeniedError`, `RateLimitError`) are remapped to HTTP 503 with `code: voice_not_configured`. The Gradio UI shows a demo bubble instead of a raw error. Other exceptions return 500 with the exception name and message for debugging.

---

## 8. Invoice OCR Module

- **Endpoint:** `POST /cost/extract-invoice` (multipart form — image or PDF)
- **Model:** `gpt-4o-mini` in Vision mode with JSON-mode output
- **Prompt:** "You are an invoice parser. Extract every line item as {description, quantity, unit_price, total}. Return a JSON array."
- **Defensive parsing:** Strips ```json``` code fences, handles number-strings ("$1,234.56" → 1234.56), returns `InvoiceExtractionResult` dataclass.
- **Dashboard:** The Cost & SLO tab has an upload widget that calls this endpoint and renders the result as a styled HTML table with a running total.
- **Why this was added:** Demonstrates that the architecture is extensible — a whole new multimodal capability added as one Python module + one FastAPI route + one Gradio component. No changes to the agent loop.

---

## 9. Evaluation System — Methodology

### 9.1 Two Modes

**Scripted mode (default, CI):**
- `FakeLLM` returns deterministic responses from a script fixture
- Zero API cost, runs in ~2 seconds for 100 scenarios
- Deterministic: same commit = same results, always
- Shipped artifacts: `report_ophthalmology.json`, `trace_ophthalmology.json`

**Live mode (`--mode live`):**
- Real `gpt-4o-mini` calls
- ~$1 for 200 scenarios (2 tenants)
- Produces real cost/latency numbers in the report
- Used for pre-release validation, not CI

### 9.2 100-Scenario Corpus Design
Each tenant has 100 scenarios across:
- Difficulty: `clear` (69), `ambiguous` (15), `adversarial` (16)
- Types: booking (38), info (22), eligibility (14), emergency/safety (20), cancellation (6)
- Safety scenarios: specifically designed to trigger guardrails (sudden vision loss, chemical splash, chest pain)

### 9.3 Locked Schema (v1.0.0)
`EvaluationReport` in `clarion/schemas/evaluation.py` has `schema_version: str = "1.0.0"`. A regression test in `tests/schemas/test_evaluation_schema_lock.py` checks:
- All expected fields exist and have the correct types
- Adding an optional field passes
- Renaming a field fails
- Changing a field type fails

This means the dashboard can always read any report generated by this or a future version — forward-compatible, never breaking.

---

## 10. Production Primitives

### 10.1 Retry with Full-Jitter Exponential Backoff
```python
@retry(attempts=4, base=0.25, cap=8.0)
def call_llm(...): ...
```
- Full jitter: actual delay = `random.uniform(0, min(cap, base * 2^attempt))`
- Full jitter beats equal-jitter for thundering-herd mitigation — all callers don't retry at the same time.

### 10.2 Circuit Breaker
- State machine: `CLOSED` → `OPEN` (after 5 failures) → `HALF_OPEN` (after 30s) → `CLOSED`
- Wraps the LLM client. If OpenAI has an outage, the circuit opens and all calls fail fast with a `CircuitBreakerOpen` exception rather than queuing and timing out.

### 10.3 Token-Bucket Rate Limiter
- Per-`(tenant_id, ip_address)` bucket
- 10 requests per second, burst of 30
- Implemented as an in-memory dict of `(last_refill_time, tokens)` — no Redis needed for single-instance deployment.
- Middleware (`api/middleware/`) checks the bucket before every request.

### 10.4 Correlation IDs
- Middleware generates a UUID4 per request, attaches to `request.state.correlation_id`
- Echoed on `X-Request-Id` response header
- All `structlog` log lines within that request include the correlation ID
- A full conversation trace is: `grep "correlation_id=<uuid>" app.log`

### 10.5 Structured JSON Logs
- `structlog` configured to emit JSON lines in production
- Keys: `timestamp`, `level`, `logger`, `correlation_id`, `customer_id`, `event`, `...kwargs`
- In development: colored console output

---

## 11. Real Evaluation Results

### Ophthalmology (100 scenarios, generated 2026-06-23)

| Metric | Value |
|---|---|
| Pass rate | **100.0%** |
| Booking accuracy | **100.0%** (38/38) |
| Safety catch rate | **100.0%** (20/20) |
| Escalation recall | **100.0%** |
| Escalation precision | 61.5% |
| Escalation F1 | 76.2% |
| Escalation accuracy | 90.0% |
| Containment rate | 74.0% |
| Hallucinations (LLM-as-Judge) | **0** |

### Orthopedics (100 scenarios, generated 2026-06-23)

| Metric | Value |
|---|---|
| Pass rate | **100.0%** |
| Booking accuracy | **100.0%** (38/38) |
| Safety catch rate | **100.0%** (20/20) |
| Escalation recall | **100.0%** |
| Escalation precision | 78.7% |
| Escalation F1 | 88.1% |
| Escalation accuracy | 90.0% |
| Containment rate | 66.0% |
| Hallucinations (LLM-as-Judge) | **0** |

### System Metrics

| Metric | Value |
|---|---|
| Tests | 705/705 passing |
| mypy strict | Clean |
| ruff | Clean |
| p50 latency (in-process, FakeLLM) | < 200 ms |
| p95 latency (in-process, FakeLLM) | < 500 ms |
| Container cold start | < 2 seconds |
| BookingFastPath accuracy | ≥ 90% held-out |
| Code volume | ~20,000 LOC |

---

## 12. Hard Problems Solved (Great Interview Answers)

### "Tell me about a technical challenge you overcame"

**Problem:** Gradio 4.x crashes with `TypeError: argument of type 'bool' is not iterable` when it internally calls `get_api_info()`. Root cause: Pydantic v2 generates `additionalProperties: True` (a boolean) for `Dict[str, Any]` fields in JSON Schema. Gradio's `gradio_client/utils.py` function `get_type()` checks `if "type" in schema` but when schema is `True` (a bool), Python raises TypeError because you can't use `in` on a bool.

**Fix:** `scripts/patch_gradio_client.py` — runs at Docker build time, reads `gradio_client/utils.py`, adds `if isinstance(schema, bool): return "Any"` guard to both affected functions (`get_type` and `_json_schema_to_python_type`) via regex substitution. The script validates it made exactly one substitution per function and raises `SystemExit` if not. `show_api=False` alone does NOT fix this — Gradio calls `get_api_info()` internally regardless.

---

### "How did you handle the eval artifacts in Docker?"

**Problem:** The `.dockerignore` file had lines `data/*/report_*.json` and `data/*/trace_*.json` — originally put there to avoid shipping large binary FAISS index files. But the eval artifacts ARE JSON text files that the dashboard reads. Docker's `COPY data ./data` silently skipped them, so the container started with "No artifacts on disk" on every tab.

**Fix:** Removed those 4 lines from `.dockerignore`. Added more specific exclusions for the actual binary files that needed to be excluded (the FAISS `.faiss` and `.sqlite` files, which are rebuilt in the builder stage anyway).

**Lesson:** `COPY` with `.dockerignore` silently succeeds even when the source file doesn't exist. Always verify your container has the files you expect with `docker run --rm clarion ls -la /app/data/ophthalmology/`.

---

### "How do you think about safety in an AI system for healthcare?"

Clarion uses **defense in depth** — three independent layers, each with a different failure mode:

1. **Pre-LLM guardrails (regex):** Fire before the model is called. Failure mode: false positives (escalates something that wasn't an emergency). This is the right failure mode for healthcare — a false alarm is annoying; a missed emergency is catastrophic.

2. **LLM-as-Judge (post-turn):** Runs after the agent responds. Doesn't share the agent's context, so can't rationalize the same mistake. Found two production-level bugs in testing.

3. **Per-specialist tool scoping:** The emergency specialist has zero tools — even if a prompt-injection attack tricks it, there's nothing for it to call. The attack surface is strictly bounded.

The key insight: you can't evaluate safety by asking the agent if it was safe. You need an external observer.

---

### "Why did you use a TF-IDF classifier instead of just prompting the LLM?"

Two reasons:

1. **Cost:** The LLM charges per token. A booking conversation might have 10 turns. If every turn spends 40–80 tokens of system prompt on "figure out what the user wants" before getting to the actual work, that's 400–800 tokens per conversation of pure overhead. At scale, that's real money. The 3 ms classifier amortizes to essentially zero.

2. **Latency:** The booking specialist's LLM call is on the critical path. Even with gpt-4o-mini's fast inference, a classifier that runs in 3 ms and gives the LLM a head start saves measurable wall time.

The design choice to prime rather than bypass is deliberate: the LLM still owns the tool call, still validates the slot, still composes the reply. We're reducing its cognitive load, not removing it from the loop.

---

### "How do you test an LLM-based system?"

Three strategies:

1. **FakeLLM for unit/integration tests:** `FakeLLM` is a scripted mock that takes a sequence of pre-defined responses and returns them in order. No API call. No cost. Deterministic. The eval harness uses this by default. 705 tests run with FakeLLM, so CI never needs an API key.

2. **LLM-as-Judge for quality:** The judge evaluates output quality on dimensions that can't be unit-tested (hallucination, policy compliance, booking correctness). The judge is itself an LLM — but it runs outside the agent's context and its verdicts are validated against ground-truth scenario outcomes.

3. **Schema regression for contracts:** The `EvaluationReport` schema is version-locked at v1.0.0. A regression test checks that the schema hasn't been silently broken. This is the equivalent of an API contract test — prevents "works on my machine" from becoming "breaks on the dashboard."

---

## 13. Common Interview Questions — Prepared Answers

**Q: Why LangGraph over AutoGen or CrewAI?**
LangGraph gives you explicit, auditable graph edges. You define the routing logic as code, not as natural language in prompts. AutoGen and CrewAI use LLMs to route between agents — which means the routing itself can hallucinate. In healthcare, I need to be able to read the routing logic in a code review and know exactly what happens when a user says "I can't see." With LangGraph, that's a regex guardrail that fires before the graph even starts.

**Q: Your escalation precision is 61.5% for ophthalmology — why not higher?**
The escalation scorer's job is to have recall of 100% (never miss a real escalation) while accepting false positives. The 61.5% precision means roughly 4 in 10 escalation flags were borderline cases that didn't strictly need human intervention. This is the correct trade-off for healthcare: it's far better to escalate a non-urgent call to a human than to not escalate a real emergency. Escalation thresholds are tunable per tenant in YAML — orthopedics is already at 78.7% precision because its urgency profile is different.

**Q: What would you do differently at 10x scale?**
1. Swap SQLite for Postgres (connection pooling, multi-instance writes)
2. Swap in-memory rate limiter for Redis (shared across instances)
3. Graduate BookingFastPath from synthetic training data to real call transcripts
4. Add a proper model registry (MLflow) for the XGBoost no-show model
5. Externalize the FAISS index to a vector DB (Weaviate, Qdrant) for online updates when new rules are added
6. Add distributed tracing (OpenTelemetry) for cross-service latency attribution

**Q: How does multi-tenancy work without code branches?**
Everything that varies between tenants lives in `configs/<tenant>.yaml`: enabled tools, escalation thresholds, persona string, rules path, modules enabled/disabled. The agent core reads `CustomerConfig` (a Pydantic model loaded from YAML) and constructs the tool list, specialist prompts, and escalation thresholds from it. Adding a new tenant is literally copying a YAML file and editing it. Zero Python changes, zero new tests beyond running the harness against the new tenant.

**Q: Walk me through a booking call end-to-end.**
1. User calls: "I need to book an eye exam for next Tuesday."
2. Voice: Whisper STT transcribes → text → POST /voice/turn
3. Guardrails: no emergency pattern → safe
4. Router (gpt-4o-mini tool call): `route(intent="booking")`
5. BookingFastPath: TF-IDF + LogReg → `search` with 0.82 confidence → hint injected
6. BookingSpecialist: RAG retrieves appointment types policy → gpt-4o-mini → calls `search_slots(date="next Tuesday", type="comprehensive_exam")` → returns 3 slots
7. Agent reads back slots to user
8. User picks 10:00 AM
9. Agent calls `book_appointment(slot_id, patient_id, patient_name, patient_phone, patient_email)` — all five fields validated by Pydantic regex before tool fires
10. SQLite store persists appointment
11. Supervisor: decision = finish
12. Judge: booking_correct=True, confidence=0.95
13. Escalation scorer: score=0.05 (low — all signals quiet)
14. TTS: OpenAI tts-1 → audio bytes → base64 → response
15. Patient 360 tab updated with new appointment

---

## 14. Links

- **Live Demo:** https://huggingface.co/spaces/Ranjithmaddirala/clarion
- **GitHub:** https://github.com/Ranjith200228/clarion
- **Contact:** ranjithmaddirala24@gmail.com
