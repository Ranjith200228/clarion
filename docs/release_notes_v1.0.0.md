# Clarion v1.0.0

*Released 2026-06-12.*

The first stable cut of Clarion — a configurable multi-agent voice
automation platform with a Sentinel trust engine. Healthcare
scheduling is the demonstration vertical; the architecture treats
vertical-specific logic as configuration, so adding another
vertical is a configuration job, not an engineering one.

This release covers the entire 19-phase build plan plus three
post-launch modules.

## Highlights

* **The engine.** A ReAct agent backed by tool-calling LLMs
  (`gpt-4o-mini` in production, `FakeLLM` in tests), wrapped in
  the Sentinel trust engine — guardrails, PHI redaction,
  LLM-as-judge, escalation scorer, audit log. Per-call traces
  with span hierarchies, token + cost accounting.
* **Two shipped customers**, ophthalmology and orthopedics, each
  with their own YAML config, structured store, rules markdown
  corpus, and 100 scenario personas.
* **Locked evaluation report schema** (`REPORT_SCHEMA_VERSION =
  "1.0.0"`) so the dashboard is a pure render layer. Every
  number it shows is in the JSON. Reports published in
  [`reports/v1.0.0/`](../reports/v1.0.0/).
* **FastAPI service + Gradio UI** — `POST /chat`, `POST /evaluate`,
  `GET /health` plus four product tabs (Live Agent, Evaluation,
  Trace Explorer, Escalations).
* **Three post-launch modules.** PMS Writeback (M1), No-Show
  Prediction (M3), Voice Layer (M5). Each opt-in per customer
  via the `modules:` YAML block.
* **Production hardening.** Structured JSON logging with
  correlation IDs, full-jitter retry, per-instance retriever
  cache, per-(customer, ip) rate limit, circuit breaker around
  the LLM client, p50/p95 SLA enforced in CI, STRIDE-shaped
  security review.

## Headline numbers (scripted mode)

100 scenarios per customer, locked-schema reports:

| Customer | Pass | Containment | Booking | Halluc. | Esc P/R | Safety |
|---|---|---|---|---|---|---|
| ophthalmology | 100% | 74.0% | 100% | 0.0% | 0.62 / 1.00 | 100% |
| orthopedics   | 100% | 66.0% | 100% | 0.0% | 0.79 / 1.00 | 100% |

* Booking accuracy and safety catch rate hit 100% on both
  customers — the agent books correctly when ground truth says
  it should, and never lets an emergency or clinical-advice
  prompt reach the model.
* Escalation recall is 100% on both — no real escalation gets
  missed.
* Hallucination is 0% on this corpus because scripted mode runs
  against `FakeLLM`. Live-mode numbers with a real OpenAI
  backend live in the Gradio app's Evaluation tab.

## Engineering posture

| Metric | Value |
|---|---|
| Unit + integration tests | 529 / 529 green |
| Load-test SLA (`-m loadtest`) | 1 / 1 green |
| mypy strict | clean across 85 source files |
| ruff lint | clean |
| Source LoC (clarion/) | ~9,800 |
| Test LoC (tests/) | ~6,400 |

Every commit on `main` was atomic (one logical change, one
test sweep, one push). The full history is the build log.

## What's new since the original spec

Three things landed beyond the original 19-phase plan, all
spec-blessed as post-launch modules:

* **Module M1 — PMS Writeback.** Convert completed
  conversations into `summary.json` + `task.json` per call,
  PHI-redacted at the writer boundary. New
  `field_extraction_accuracy` metric folds into the locked
  report.
* **Module M3 — No-Show Prediction.** Synthetic dataset
  generator, XGBoost trainer with stratified 5-fold CV,
  predictor with low/medium/high risk bands and a
  feature-column drift guard. New `no_show_roc_auc` +
  `no_show_top_decile_lift` metrics.
* **Module M5 — Voice Layer.** STT + TTS protocols with
  faster-whisper / OpenAI prod implementations and
  echo / sine-wave test stubs. `VoiceOrchestrator` chains
  STT → agent → TTS reusing the same conversation engine that
  powers `/chat`. New `POST /voice/turn` endpoint.

The Phase 18 production hardening package landed alongside —
seven commits covering observability, resilience, perf, and
security.

## Compatibility

The marketing version (`1.0.0`) is independent of the wire
contracts. Within the 1.x line:

| Contract | Pinned at | Module |
|---|---|---|
| `EvaluationReport`, `TraceReport` | `1.0.0` | `clarion/schemas/evaluation.py` |
| `ConversationSummary`, `PmsTaskWriteback` | `1.0.0` | `clarion/schemas/modules.py` |
| `NoShowPrediction`, `NoShowModelMetadata` | `1.0.0` | `clarion/schemas/modules.py` |
| `VoiceTurnRequest`, `VoiceTurnResponse`, `TranscriptionResult`, `AudioMetadata` | `1.0.0` | `clarion/schemas/modules.py` |

Additive changes (new optional field, looser bounds) keep
these versions stable through the 1.x line. Removals or
renames will bump the schema version independently of the
marketing release.

## Known limitations

See [`docs/security_review.md`](security_review.md) for the
full audit. Headline gaps:

1. No authentication middleware — the demo trusts the caller's
   identity.
2. Regex PHI redactor is best-effort; production needs an NLP
   scrubber.
3. Rate limit is in-process; multi-replica deploys need
   Redis-backed buckets.
4. Audit + trace logs land on the container's writable
   filesystem, not tamper-evident storage.
5. The OpenAI Enterprise tier is operationally enforced, not
   protocol enforced.
6. No HIPAA paperwork; this is a healthcare-shaped portfolio
   piece, not a HIPAA-certified product.

None are technically out of reach — they're the
production-readiness work that doesn't fit in a portfolio demo.

## Acknowledgements

The spec for this build (the 19-phase plan, the LOCK REPORT
SCHEMA rule, the recruiter test as Definition of Done, the
healthcare-as-demonstration framing, the modules-must-stay-isolated
contract) shaped every commit.

## Upgrade path

This is the first release; there's nothing to upgrade from.
Future 1.x point releases will carry their own notes here under
[`docs/release_notes_*.md`](.) and an entry in
[`CHANGELOG.md`](../CHANGELOG.md).
