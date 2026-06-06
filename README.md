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

Phase 4 complete — five validated tools (`search_slots`, `book_appointment`,
`cancel_appointment`, `check_eligibility`, `create_pms_task`) sit on top of
the dual data pipeline behind a registry that enforces per-customer
`enabled_tools` (e.g. orthopedics can't call `cancel_appointment` because
its YAML says so). Tools never raise to the agent — they return structured
`ok=False` results so the LLM can recover without exception handling.

## Build phases

| Phase | Scope | Status |
| ----- | ----- | ------ |
| 1     | Foundation (Poetry, Ruff, Pytest, Docker, CI) | ✅ complete |
| 2     | Multi-tenant config system                    | ✅ complete |
| 3     | Dual data pipeline (RAG + SQLite)             | ✅ complete |
| 4     | Schemas + mocked tools                        | ✅ complete |
| 5     | Text agent MVP (ReAct)                        | pending |
| 6     | Guardrails (emergency / clinical / PHI)       | pending |
| 7     | Observability (tokens, latency, cost, traces) | pending |
| 8     | FastAPI service                               | pending |
| 9     | Simulation harness                            | pending |
| 10    | Sentinel trust engine (LLM-as-judge)          | pending |
| 11    | Escalation engine                             | pending |
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
