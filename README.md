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

Phase 1 — foundation only. The project structure, tooling (Poetry, Ruff,
Pytest, MyPy, pre-commit), Docker, and CI are in place. Functional modules
land in subsequent phases.

## Build phases

| Phase | Scope | Status |
| ----- | ----- | ------ |
| 1     | Foundation (Poetry, Ruff, Pytest, Docker, CI) | in progress |
| 2     | Multi-tenant config system                    | pending |
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
