# Developer Guide

Onboarding for engineers extending Clarion. Assumes you've cloned the
repo and have Python 3.11+ available.

For end-to-end deployment instructions see
[`deployment_guide.md`](./deployment_guide.md).
For the architecture overview see [`../README.md`](../README.md) and the
[architecture diagram](./architecture.png).

---

## 1. Local development setup

```bash
# Install Poetry if you don't have it
pip install --user poetry

# Install runtime + dev deps; add --with ui for the Gradio app
poetry install --with dev --with ui

# Activate the venv
poetry shell
```

Required env vars for the full local stack:

```bash
export OPENAI_API_KEY=sk-...           # for live LLM (FakeLLM works without)
export CLARION_CUSTOMER=ophthalmology  # default tenant for CLI
export CLARION_API_URL=http://localhost:8000   # Gradio -> FastAPI
```

`.env` files at repo root are honored via `python-dotenv` (already a
pydantic-settings dep).

---

## 2. Repo tour

| Path | What lives there |
|---|---|
| `clarion/config/` | Pydantic `Settings` + per-customer YAML loader |
| `clarion/pipelines/` | Structured (SQLite) + unstructured (markdown → chunks) ingest |
| `clarion/rag/` | FAISS retriever + TF-IDF + OpenAI embedders |
| `clarion/agents/` | ReAct loop, OpenAI client, FakeLLM, prompt builder |
| `clarion/tools/` | Five shipped tools + registry that enforces `enabled_tools` |
| `clarion/sentinel/` | Guardrails, PHI redactor, judge, escalation scorer |
| `clarion/observability/` | Span tracer, cost calculator, JSONL writer |
| `clarion/simulator/` | 100-persona generator + harness |
| `clarion/evaluation/` | `runner.py`, `metrics.py`, `reporter.py`, `trace_report.py` |
| `clarion/eval/` | Canonical CLI: `python -m clarion.eval --customer X` |
| `clarion/schemas/` | All Pydantic wire shapes (locked report contract lives here) |
| `api/` | FastAPI service (`/chat`, `/evaluate`, `/health`) |
| `gradio_app/` | Gradio Blocks UI with 4 tabs + customer switcher |
| `configs/` | Per-customer YAML files |
| `data/seeds/` | Per-customer structured-store seed JSON |
| `data/rules/<customer>/` | Per-customer rules markdown corpus |
| `data/personas/` | 100 synthetic scenarios per customer |
| `tests/` | pytest suite mirroring the source tree |
| `scripts/` | Build / health / render-architecture helpers |
| `deploy/` | Cloud Run / Render / Fly.io manifests |

---

## 3. Daily commands

```bash
# Run the full test suite
poetry run pytest -q

# Type-check
poetry run mypy

# Lint + format
poetry run ruff check .
poetry run ruff format .

# Regenerate FAISS indices for both shipped customers
poetry run python -m clarion.pipelines.ingest all ophthalmology
poetry run python -m clarion.pipelines.ingest all orthopedics

# Regenerate personas
poetry run python -m clarion.simulator.cli generate all

# Run the evaluation harness end-to-end
poetry run python -m clarion.eval --customer ophthalmology --out reports/
poetry run python -m clarion.eval --customer orthopedics --out reports/

# Launch the FastAPI service locally
poetry run python -m api.app

# Launch the Gradio UI (separate shell)
poetry run python -m gradio_app

# Or both via docker-compose
docker compose up
```

---

## 4. Adding a new tool

Five-step recipe; the existing five shipped tools follow this exact
template.

1. **Schema** — add `XInput` and `XOutput` Pydantic models to
   `clarion/schemas/tools.py`. `XOutput` should subclass `ToolOutput`
   so it has `ok` + `error`. Use `extra="forbid"` so unknown fields
   are rejected at the LLM boundary.
2. **Implementation** — create `clarion/tools/x.py` exporting an
   `XTool` class with `name`, `input_model`, `output_model` class
   attributes and a `run(input, ctx)` method. Wrap any I/O in
   `run_with_retry(...)` so transient SQLite errors are retried. Tools
   never raise to the agent; failures become `ok=False`.
3. **Register** — add to `clarion/tools/registry.py` `_REGISTRY` dict
   and to the `ToolName` Literal in `clarion/config/schema.py`.
4. **Test** — `tests/tools/test_x.py` covering happy path, every
   `ok=False` path, and Pydantic-validation rejection of malformed
   input. Use the fixtures from `tests/tools/conftest.py`.
5. **Wire into customer YAML** — operators add the tool name to
   `enabled_tools` in `configs/<customer>.yaml`. No agent-code change
   needed; the registry honors the per-customer allowlist.

---

## 5. Adding a new customer (FDE flow)

The point of Clarion is that this is a one-day job, not an engineering
project. Steps:

1. **Discovery** — fill out a `docs/discovery_<customer>.md` (see the
   sample `docs/discovery.md`) so the next person knows why each YAML
   choice was made.
2. **YAML** — drop `configs/<customer_id>.yaml`. Required fields:
   `customer_id`, `display_name`, `specialties`, `enabled_tools`,
   `escalation`, `languages`, `rules_path`, `agent_persona`. The
   schema rejects unknown fields, so you'll see typos immediately.
3. **Seed** — write `data/seeds/<customer_id>.json` mirroring the
   shape of `data/seeds/ophthalmology.json`. ~3 providers + ~10 slots
   + ~4 eligibility records is enough for the demo.
4. **Rules** — write 5-7 markdown files in
   `data/rules/<customer_id>/` covering appointment types,
   new-patient intake, payer policy, cancellation policy, emergency
   escalation. The chunker splits on H2/H3 so structure matters.
5. **Personas** — `python -m clarion.simulator.cli generate <customer_id>`
   produces 100 synthetic scenarios using the customer's appointment
   types and tool allowlist. The generator reads from `configs/` and
   needs no per-customer code.
6. **Ingest + evaluate**:
   ```bash
   python -m clarion.pipelines.ingest all <customer_id>
   python -m clarion.eval --customer <customer_id>
   ```
   Read the resulting `report_<customer_id>.json` — anything below
   the existing customers' numbers is a tunable.

**No agent code changes.** If you find yourself editing
`clarion/agents/` to handle a customer's quirk, it belongs in their
YAML or rules corpus instead.

---

## 6. Adding a new metric

Same one-way dependency graph the Phase 13 spec mandates:

1. Add the field to `EvaluationMetrics` in
   `clarion/schemas/evaluation.py`. Additive = no version bump.
2. Compute the value in `clarion/evaluation/metrics.py`. Pull from
   `HarnessReport.results` or `traces.jsonl` summaries.
3. Wire it into the `EvaluationMetrics(...)` construction at the
   bottom of `_metrics_for`.
4. Surface it in the Quality tab in
   `gradio_app/tab_quality.py` (UI renders only — no math).
5. Test in `tests/evaluation/test_metrics.py`.

If the metric is a derived breakdown (e.g. outcome distribution),
also add a top-level field on `EvaluationReport` and populate it in
`reporter.py` so the UI continues to read directly.

---

## 7. Test conventions

- One test file per module, mirroring the source layout.
- pytest fixtures live in the nearest `conftest.py`.
- Unit tests use `tmp_path` for any filesystem write; never write to
  `data/` from a test.
- E2E tests use `FakeLLM` with a scripted response list. CI must run
  without `OPENAI_API_KEY`.
- Mypy strict is required for everything in `clarion/`. Per-file
  `# type: ignore` is acceptable; module-wide ignores are not.

---

## 8. CI pipeline

`.github/workflows/ci.yml` runs on every push and PR:

1. **Lint** — `ruff check . && ruff format --check .`
2. **Type-check** — `mypy`
3. **Tests** — `pytest -q` (against Python 3.11 + 3.12)
4. **Phase 15 acceptance** — `docker build --target test -t clarion:test .`
   runs the suite inside the container image.

A green CI is the only merge gate.

---

## 9. Common pitfalls

- **PHI in commits.** Real customer data must never land in
  `data/seeds/` or `data/rules/`. The repo ships synthetic data only.
- **Forgetting `--with ui`** when installing — the Gradio app won't
  start without it.
- **Stale FAISS indices** after changing rules markdown — rerun
  `python -m clarion.pipelines.ingest`.
- **`extra="forbid"` rejection on tool inputs** when the LLM emits a
  field that doesn't exist — that's the contract working; tighten the
  prompt or the input schema.

---

## 10. Where to look next

- The architecture diagram: [`architecture.png`](./architecture.png) /
  [`architecture.mmd`](./architecture.mmd).
- The locked report contract:
  `clarion/schemas/evaluation.py` (`schema_version`).
- The customer-discovery template: [`discovery.md`](./discovery.md).
- The deployment guide: [`deployment_guide.md`](./deployment_guide.md).
