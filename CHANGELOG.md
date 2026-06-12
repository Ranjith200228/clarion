# Changelog

All notable changes to **Clarion** are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project follows [Semantic Versioning](https://semver.org/).

## [1.0.0] — 2026-06-12

The first stable release. Clarion is a configurable multi-agent
voice automation platform with a Sentinel trust engine, ships with
two healthcare-vertical demo customers (ophthalmology, orthopedics),
and clears the recruiter test from the spec.

### Added — engine
- Multi-tenant `CustomerConfig` schema + YAML loader; two shipped
  customers under `configs/`.
- Structured + unstructured data pipelines (Phase 3): `StructuredStore`
  for slots / providers / eligibility; chunker + TF-IDF / OpenAI
  embedder + FAISS retriever for rules.
- ReAct agent (Phase 5) with `FakeLLM` + `OpenAIClient`; rolling
  transcript, persona-aware system prompts, deterministic tool
  dispatch.
- Tool registry (Phase 4): `search_slots`, `book_appointment`,
  `cancel_appointment`, `check_eligibility`, `create_pms_task` —
  each defined by a Pydantic input/output schema.
- Sentinel trust engine (Phases 6–11): guardrails, PHI redactor,
  LLM-as-judge, escalation scorer, audit log writer.
- Trace observability (Phase 10): hierarchical span recorder,
  cost calculator (gpt-4o-mini pricing), JSONL writer.
- Simulator harness (Phase 12): 100-scenario corpus per customer,
  scripted + live modes, per-scenario `HarnessResult`.

### Added — evaluation + reporting
- Phase 12 metrics: pass rate, containment, booking accuracy,
  hallucination rate, escalation P/R/F1, safety catch rate,
  avg turns, cost per request, latency stats.
- Phase 13 **locked report schema**:
  `REPORT_SCHEMA_VERSION = "1.0.0"`,
  `TRACE_SCHEMA_VERSION = "1.0.0"`. Additive changes only — every
  subsequent feature respected the lock.
- `report_<customer>.json` + `trace_<customer>.json` per run.
- Pre-aggregated UI feeds folded into the report (outcome
  distribution, escalation reason frequency, escalated scenario
  ids) so the dashboard stays a pure render layer.

### Added — UI + API
- Phase 8: FastAPI service with `POST /chat`, `POST /evaluate`,
  `GET /health`; session manager with conversation continuity.
- Phase 14: Gradio Blocks UI (`gradio_app/`) — Live Agent,
  Evaluation, Trace Explorer, Escalations tabs + customer
  switcher. Zero metric computation in the UI; tabs render
  pre-aggregated feeds straight from the locked report.

### Added — deployment
- Phase 15: multi-stage Dockerfile (builder → test → runtime),
  non-root UID 1000, FAISS indices pre-built into the image so
  cold start is under 2 s.
- Phase 16 deployment manifests: HuggingFace Spaces
  (`huggingface/README.md`), Cloud Run (`deploy/cloudrun.yaml`),
  Render (`deploy/render.yaml`), Fly.io (`deploy/fly.toml`).
- Phase 17 documentation: deployment guide, developer guide,
  architecture diagram, discovery doc.

### Added — Module M1: PMS Writeback
- `clarion/modules/pms_writeback/` — `Extractor` Protocol,
  `HeuristicExtractor`, `PmsWritebackWriter` producing
  `summary.json` + `task.json` per conversation at
  `<data_dir>/<customer>/pms_writeback/<conv>/`.
- PHI redaction at the writer boundary — every string leaf
  flows through `clarion.sentinel.phi.redact` before
  serialization.
- New `EvaluationMetrics.field_extraction_accuracy` (additive,
  schema 1.0.0 preserved); null when the module is disabled.

### Added — Module M3: No-Show Prediction
- `clarion/modules/no_show_prediction/` — synthetic dataset
  generator, XGBoost trainer with 5-fold stratified CV,
  `NoShowPredictor` with risk-band mapping
  (low / medium / high).
- Feature-column drift guard refuses to load a bundle whose
  persisted `feature_columns` disagrees with the dataset
  module's current `FEATURE_COLUMNS`.
- New `EvaluationMetrics.no_show_roc_auc` +
  `no_show_top_decile_lift` (both additive, schema 1.0.0
  preserved); held-out eval seed differs from training seed
  for honest out-of-fold measurement.
- Adds `xgboost ^2.1` to core deps.

### Added — Module M5: Voice Layer
- `clarion/modules/voice/` — `TranscriberProtocol` +
  `SpeakerProtocol` with two implementations each:
  `FasterWhisperTranscriber` / `EchoTranscriber`,
  `OpenAITtsSpeaker` / `SineWaveSpeaker`.
- `VoiceOrchestrator.turn(agent, audio, ...) -> VoiceTurnResponse`
  chaining STT → agent → TTS with per-stage latencies.
- `POST /voice/turn` endpoint; rejects when no orchestrator is
  injected (503 `voice_not_configured`).
- faster-whisper lazy-imported so the ~1 GB dep cost is paid
  only by deployments that enable voice.

### Added — Production hardening (Phase 18)
- Structured JSON logging (`clarion.observability.logging`) with
  context-bound correlation IDs and request-scoped middleware.
- Retry decorator with full-jitter exponential backoff
  (`clarion.resilience.retry`); wraps the OpenAI network
  boundary.
- Per-instance LRU cache on `Retriever.retrieve` (default 64
  entries) with inspectable hit/miss/size stats.
- Token-bucket rate limiter keyed by (customer_id, ip) on
  `/chat` + `/voice/turn`; per-tenant isolation; 429 with
  Retry-After.
- Circuit breaker (`clarion.resilience.circuit_breaker`) wrapping
  the OpenAI client; 5 failures trip a 30 s cooldown; HALF_OPEN
  probe on recovery.
- Load-test harness: locust profile for live deployments,
  in-process burst test enforcing p50 < 200 ms / p95 < 500 ms
  via the `loadtest` pytest marker.
- Security review (`docs/security_review.md`) — STRIDE-shaped
  audit naming both controls and gaps.

### Locked contracts
The following contracts are stable across the 1.x line. Breaking
changes will bump these versions independently of the marketing
release version:

| Contract | Version | Lives in |
|---|---|---|
| `EvaluationReport` | `1.0.0` | `clarion/schemas/evaluation.py` |
| `TraceReport` | `1.0.0` | `clarion/schemas/evaluation.py` |
| `ConversationSummary`, `PmsTaskWriteback` | `1.0.0` | `clarion/schemas/modules.py` |
| `NoShowPrediction`, `NoShowModelMetadata` | `1.0.0` | `clarion/schemas/modules.py` |
| `VoiceTurnRequest`, `VoiceTurnResponse`, `TranscriptionResult`, `AudioMetadata` | `1.0.0` | `clarion/schemas/modules.py` |

### Verified
- 529 unit + integration tests green (mypy strict, ruff clean).
- 1 load-test SLA check green under the `loadtest` marker.
- Doc-contract tests enforce the report-schema lock + key README
  sections.
- Multi-stage Docker build green; non-root runtime.

### Known gaps
See [`docs/security_review.md`](docs/security_review.md) for the
full list. Summary: identity middleware, NLP-quality PHI scrubber,
Redis-backed multi-replica rate limits, tamper-evident audit
storage, penetration test, HIPAA paperwork. None of these are out
of reach — they're the production-readiness work that doesn't fit
in a portfolio demo.

[1.0.0]: https://github.com/Ranjith200228/clarion/releases/tag/v1.0.0
