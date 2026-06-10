# Deploying Clarion

Phase 16 ships one container image that runs unchanged on four targets:

| Target | When to use | Cost |
|---|---|---|
| **Hugging Face Gradio Space** | Public demo for portfolios / recruiters | Free (cpu-basic) |
| Cloud Run | Production-grade auto-scaling | Pay-per-request |
| Render | One-click Blueprint deploy | Starter tier ~$7/mo |
| Fly.io | Multi-region, scale-to-zero | Hobby tier free |

The same Phase 15 Docker image works on all of them. Per-target
manifests live under `deploy/` and at `huggingface/README.md`.

---

## Prerequisites

1. **Build the image once locally** to verify everything still works:
   ```bash
   docker compose build
   docker compose up        # sanity check on :7860 and :8000/docs
   ```
2. **Push the image to a registry** the target reads from. Cloud Run
   uses Artifact Registry; Render + Fly.io + HF Spaces all build from
   the Dockerfile in the repo, no manual push needed.
3. **Have your `OPENAI_API_KEY` ready.** All targets accept it as a
   secret env var. Without it the Live Agent tab falls back to
   guardrail-only responses; the read-only tabs still work because
   they consume the prebuilt JSON reports.

---

## Primary: Hugging Face Gradio Space

1. Create a new Space at https://huggingface.co/new-space:
   - **SDK**: Docker
   - **Visibility**: Public
2. Clone the new empty Space repo to your machine.
3. Copy this project's tracked files into it (or set the Space repo to
   mirror this GitHub repo).
4. Copy the contents of `huggingface/README.md` from this repo into
   the Space's **root** `README.md` so HF reads the YAML frontmatter
   from there (`sdk: docker`, `app_port: 7860`, etc.).
5. In the Space's **Settings → Variables and secrets**, set:
   - `OPENAI_API_KEY` (secret)
   - `CLARION_MODEL` (variable, optional — defaults to `gpt-4o-mini`)
6. Push to the Space's `main` branch. HF builds the Docker image and
   starts the container; `scripts/serve_all.sh` brings up FastAPI on
   `:8000` (loopback) and Gradio on `:7860` (HF's `app_port`).
7. Once the Space says **Running**, the public URL is your demo.

Phase 16 acceptance: *Public Hugging Face URL works.*

---

## Cloud Run

```bash
gcloud config set project <PROJECT>

# Build + push to Artifact Registry.
gcloud builds submit --tag <REGION>-docker.pkg.dev/<PROJECT>/clarion/clarion:v1

# Store the secret.
gcloud secrets create clarion-openai-key --replication-policy=automatic
echo -n "sk-..." | gcloud secrets versions add clarion-openai-key --data-file=-

# Edit deploy/cloudrun.yaml — replace <PROJECT>, <REGION>, <IMAGE_TAG>,
# and <SECRET_NAME> (set <SECRET_NAME>=clarion-openai-key).

# Apply.
gcloud run services replace deploy/cloudrun.yaml --region <REGION>
```

The service URL Cloud Run prints is the public endpoint. Health probes
hit `/` on port 7860 (the Gradio root); the API is reachable from the
Gradio process on `127.0.0.1:8000` thanks to `scripts/serve_all.sh`.

---

## Render

1. Push this repo to GitHub.
2. In the Render dashboard: **New → Blueprint**, point at the repo.
3. Render reads `deploy/render.yaml` (rename or symlink to
   `render.yaml` at the repo root if Render's resolver requires it).
4. Set `OPENAI_API_KEY` in the dashboard (`sync: false` keeps it out
   of git).
5. Render builds + deploys automatically on every push to `main`.

---

## Fly.io

```bash
fly launch --copy-config --no-deploy   # one-time, picks region etc.

# Adjust deploy/fly.toml — set ``app = "your-globally-unique-name"``.

fly secrets set OPENAI_API_KEY=sk-...

fly deploy
```

`fly.toml` has `auto_stop_machines = "stop"` + `min_machines_running = 0`,
so the demo scales to zero between visitors.

---

## Pre-built FAISS index — no manual build at deploy time

The Phase 15 `builder` stage runs `scripts/build_indices.sh` and bakes
the per-customer FAISS index + SQLite store into the image:

```
/app/data/ophthalmology/structured.sqlite
/app/data/ophthalmology/rules.faiss
/app/data/ophthalmology/rules_meta.json
/app/data/orthopedics/structured.sqlite
/app/data/orthopedics/rules.faiss
/app/data/orthopedics/rules_meta.json
```

Operators **do not** need to build the index manually on the deploy
target. First `/chat` request lands on a fully warm RAG pipeline.

---

## Synthetic demo data ships with the image

`data/seeds/<customer>.json` and `data/personas/<customer>.json` are
checked into the repo and copied into the runtime image. The Gradio
UI's read-only tabs (Quality, Escalations, Trace Explorer) work
out-of-the-box because pre-built `report_<customer>.json` +
`trace_<customer>.json` files can be generated via
`python -m clarion.eval --customer all` before the deploy push.

If you want fresh reports, run before deploying:

```bash
poetry run python -m clarion.eval --customer all --out data
```

This writes `report_<customer>.json` + `trace_<customer>.json` under
`data/<customer>/` which the image copies in. The UI reads them on
first load.

---

## Environment variables — the full list

| Var | Default | Set per target |
|---|---|---|
| `OPENAI_API_KEY` | (empty) | HF secret / Cloud Run secret / Render `sync: false` / `fly secrets set` |
| `CLARION_MODEL` | `gpt-4o-mini` | Optional override |
| `CLARION_DATA_DIR` | `/app/data` | Baked into Dockerfile; rarely overridden |
| `CLARION_CONFIG_DIR` | `/app/configs` | Baked into Dockerfile |
| `CLARION_API_HOST` | `127.0.0.1` | Internal loopback for in-container API |
| `CLARION_API_PORT` | `8000` | Internal port |
| `CLARION_API_URL` | derived from above | Override for advanced topologies |
| `GRADIO_HOST` | `0.0.0.0` | Always public bind |
| `GRADIO_PORT` | `7860` | HF Spaces app_port |

Everything else is config-driven through the per-customer YAMLs in
`configs/`. No code change needed to deploy a new customer — see the
project README for the multi-tenant story.
