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

Phase 2 complete — Clarion now boots from a customer YAML. Two demonstration
customers are shipped (ophthalmology and orthopedics); agent code (lands in
Phase 5) reads `CustomerConfig` and behaves accordingly with no per-customer
branching.

## Build phases

| Phase | Scope | Status |
| ----- | ----- | ------ |
| 1     | Foundation (Poetry, Ruff, Pytest, Docker, CI) | ✅ complete |
| 2     | Multi-tenant config system                    | ✅ complete |
| 3     | Dual data pipeline (RAG + SQLite)             | pending |
| 4     | Schemas + mocked tools                        | pending |
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
│   ├── rules/             # unstructured rules corpora (gitignored)
│   └── personas/          # synthetic patient personas (gitignored)
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
