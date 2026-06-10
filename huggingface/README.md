---
title: Clarion — Configurable Multi-Agent Voice Automation
emoji: 🛎️
colorFrom: indigo
colorTo: gray
sdk: docker
app_port: 7860
suggested_hardware: cpu-basic
suggested_storage: small
pinned: false
license: mit
short_description: Multi-tenant healthcare scheduling agent with a Sentinel trust engine
tags:
  - agents
  - rag
  - evaluation
  - observability
  - fastapi
  - gradio
  - fde
---

# Clarion on Hugging Face Spaces

This Space ships the full Clarion stack — FastAPI agent backend +
Gradio product UI — as a single Docker SDK Space.

The image is the same one local `docker compose up` builds (see the
project [Dockerfile](https://github.com/Ranjith200228/clarion/blob/main/Dockerfile)).
`scripts/serve_all.sh` starts FastAPI on loopback `:8000` and Gradio
on `:7860` (HF's `app_port`).

## How this Space is wired

This file lives at `huggingface/README.md` in the repo for reference.
When you create the Space, **copy this file's contents into the Space's
own root `README.md`** so HF reads the frontmatter from there. The
Space's repo otherwise tracks the project repo.

## Secrets

Set in the Space's **Settings → Variables and secrets**:

| Name | Required? | Notes |
|---|---|---|
| `OPENAI_API_KEY` | Required for live LLM | Without it the Live Agent tab falls back to guardrail-only responses. |
| `CLARION_MODEL` | Optional | Defaults to `gpt-4o-mini`. |

`CLARION_DATA_DIR`, `CLARION_CONFIG_DIR`, `CLARION_API_URL`,
`GRADIO_HOST`, and `GRADIO_PORT` are baked into the Dockerfile and
don't need to be set on the Space.

## What you'll see

When the Space boots it shows the Gradio Blocks UI with four tabs:

- **Live Agent** — `gr.ChatInterface` calling the in-Space FastAPI
- **Quality Metrics** — reads `data/<customer>/report_<customer>.json`
- **Escalations** — reads the same report
- **Trace Explorer** — reads `data/<customer>/trace_<customer>.json`

A customer-switcher dropdown toggles between the shipped tenants
(ophthalmology, orthopedics).

## Resource sizing

The default **cpu-basic** tier (16 GB RAM, 2 vCPU) is enough — the
FAISS indices for both shipped customers fit in <100 MB and the agent's
peak memory is well under 1 GB. Storage stays under 1 GB.

## License

MIT.
