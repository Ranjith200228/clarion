# IMPLEMENTATION_PLAN.md — how Clarion becomes mission-control

*Phase 0 deliverable. Names the components, the build order, the
shared visual system, the data sources, and the deployment impact.
Implementation does not start until this plan is acknowledged.*

---

## 1. Guiding constraints (carried forward from the vision brief)

1. Gradio Blocks stays. We use its theming API, `gr.HTML`, `gr.Plot`,
   and aggressively styled `gr.Markdown` — but no React/Next.js.
2. FastAPI stays. No new endpoint is required for the v2 experience;
   every new view reads from existing routes or on-disk artifacts.
3. LangGraph, Sentinel, FAISS, Whisper/TTS, observability, locked
   schemas — all preserved unchanged.
4. The deployment pipeline (Docker multi-stage → HF Spaces →
   `scripts/serve_all.sh` → port 7860) stays. New CSS / new tabs
   land in the existing image.

---

## 2. The shared visual system

The single biggest change is **introducing a visual language**.
Everything below is built on top of it.

### 2.1 Theme

`gradio_app/theme.py` — a `gr.themes.Base` subclass:

- **Mode**: dark default with a light toggle (recruiter-friendly)
- **Brand**: deep navy `#0B1220` background, near-white text
  `#E5E7EB`, single accent `#06B6D4` (cyan) for primary, `#10B981`
  (emerald) for "healthy", `#F59E0B` (amber) for "warning",
  `#EF4444` (red) for "critical"
- **Typography**: Inter for UI, JetBrains Mono for spans / IDs /
  cost values
- **Density**: compact — chips, badges, and tiles cluster tightly
  the way Datadog does
- **Motion**: ease-out 150 ms for hover; 250 ms for tab fade-in;
  no decorative animation

### 2.2 Component primitives (`gradio_app/components.py`)

Pure-function builders that return `gr.HTML` blocks. Each takes
typed inputs and returns a rendered chip / tile / gauge / etc. No
state, no callbacks — these are render helpers we use everywhere.

| Builder | Purpose |
|---|---|
| `kpi_tile(label, value, delta, status)` | 120×96 px tile; large value, small delta, status border-left color |
| `status_badge(state)` | small pill: `healthy` / `warning` / `critical` / `unknown` |
| `trust_gauge(score, threshold)` | inline SVG semicircle gauge 0-1 |
| `signal_bar(name, value, weight)` | horizontal weighted bar for one escalation signal |
| `latency_ring(stage, ms, target_ms)` | small ring chart for STT / Agent / TTS |
| `tenant_card(customer_id, health, last_run_at)` | left-rail nav item |
| `incident_row(ts, severity, tenant, summary)` | one row in the incidents stream |
| `agent_node(name, state, ms, cost)` | one node in the agent-flow SVG |
| `cost_chip(usd, period)` | monospace cost pill |
| `mono(text)` | JetBrains Mono span (trace IDs, conversation IDs) |

Every builder is unit-tested via snapshot tests so the visual
language can't drift silently.

### 2.3 CSS strategy

- A single `gradio_app/style.css` is loaded by `gr.Blocks(css=...)`.
- Three layers of CSS:
  1. **Tokens** — CSS variables for color, spacing, radius, type
     scale. One source of truth.
  2. **Primitives** — `.kpi-tile`, `.status-badge`, `.gauge`,
     `.tenant-card`, etc. Class-based, never inline.
  3. **Layouts** — utility classes for the mission-control grid.
- Custom CSS never overrides Gradio's internals (defensive against
  Gradio version bumps). We style INSIDE our `gr.HTML` blocks; we
  do not selector-hunt for `.gradio-button`.

---

## 3. The new application shell

`gradio_app/app.py` is restructured around a three-zone layout:

```
┌────────────────────────────────────────────────────────────────┐
│ TOP STRIP                                                       │
│   ◐ Clarion    v1.1.0  · prod  · healthy   [Customer ▼]  [☼]   │
│                                                                 │
│   ┌─KPI─┐ ┌─KPI─┐ ┌─KPI─┐ ┌─KPI─┐ ┌─KPI─┐ ┌─KPI─┐ ┌─KPI─┐ ┌─KPI─┐│
│   │ 100%│ │70.0%│ │ $0.0│ │ 100%│ │  0  │ │ 1.42│ │ 0.83│ │  3  ││
│   │Pass │ │Cont.│ │Cost │ │Safe │ │Err  │ │Trust│ │Esc-P│ │Tnts ││
│   └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘ └─────┘│
├──────────┬──────────────────────────────────────────────────────┤
│ LEFT NAV │ MAIN CANVAS                                          │
│          │                                                      │
│ ◇ Mission│   (each view renders here)                           │
│ ◇ Trust ★│                                                      │
│ ◇ Agents │                                                      │
│ ◇ Voice  │                                                      │
│ ◇ Quality│                                                      │
│ ◇ Escal. │                                                      │
│ ◇ Trace  │                                                      │
│ ◇ Health │                                                      │
│ ◇ Cost   │                                                      │
└──────────┴──────────────────────────────────────────────────────┘
```

- Top strip + left nav are `gr.HTML` blocks rebuilt by
  `refresh_global(...)` on every customer change.
- Main canvas is a `gr.Tabs` with hidden tab labels; left-nav clicks
  switch tabs (Gradio supports this via `change` callbacks).
- ★ = primary, lit cyan; everything else lit muted.

---

## 4. The nine views (and their data sources)

In **build order**:

### 4.1 Mission Control (home)

The 5-second-recruiter view. Defaults open here.

- Top strip KPIs (above).
- Tenant table: 1 row per customer, columns =
  `customer_id`, `pass_rate`, `containment`, `safety_catch`,
  `last_trace_at`, status badge.
- Recent escalations stream (last 10 from any customer's
  `EvaluationReport.escalated_scenario_ids`).
- Recent emergencies stream (audit log filtered to
  `emergency_intent_classified`).

Data: existing `report_<customer>.json` + `audit_<customer>.jsonl`.
No new endpoints.

### 4.2 Sentinel Operations Center (★ hero)

The trust-engine view. Split into 4 panels.

- **Live Trust Gauge** — composite escalation score across the
  last 100 turns. SVG semicircle, 0-1 scale, threshold line.
- **Signal Breakdown** — five horizontal weighted bars
  (`low_confidence`, `repeated_clarification`, `rule_conflict`,
  `frustration`, `unsupported_request`). Bar length = raw signal,
  bar color = signal × weight contribution.
- **Judge Confidence Trail** — last 50 turns, scatter of judge
  confidence vs. our composite scorer. Inline annotation:
  "scorer agreement = 87%".
- **PHI Redaction Counter + Audit Tail** — rolling count of
  redactions today + last 10 audit-log lines, PHI-already-redacted.

Data: `EvaluationReport.metrics.escalation_*` + Tracer span
attributes + `audit_<customer>.jsonl`.

### 4.3 Agent Flow (★ hero)

The multi-agent view. One animated SVG plus a trace pop-out.

- **Live Flow Diagram** — SVG nodes for
  `Patient → Router → {Booking|Eligibility|Info|Cancel|Emergency}
  → Tools (search/book/eligibility/task) → Sentinel → Response`.
- Each node carries:
  - state badge (`idle` / `active` / `done` / `escalated`)
  - ms spent
  - cost USD (when applicable)
  - tool count when fanout
- **Turn Inspector** — when a user types in the chat (right side
  of the page), the diagram lights up the path that was actually
  taken; clicking a node opens its span detail.
- The chat itself stays — it's just no longer the centerpiece. It's
  the input that drives the flow.

Data: live `TraceReport` entries + the new `last_turn_spans` on
the agent.

### 4.4 Voice Intelligence (★ hero)

- **Waveform** — live during recording; idle when stopped.
  `gr.Plot` from a small librosa-free numpy implementation
  (`scipy.signal.spectrogram` is allowed; we have it transitively).
- **Live Transcript** — token-by-token from the existing
  TranscriptionResult.text; per-token color tied to confidence
  when available.
- **Emotion Meter** — bar across 6 emotions (calm, anxious,
  confused, frustrated, urgent, distressed). Derived from
  existing `frustration.py` heuristics + simple sentiment scoring
  on the transcript (NLTK VADER, already a transitive dep) — no
  new model load.
- **Frustration Trace** — turn-over-turn line chart.
- **Escalation Prediction** — single-number "this caller will be
  escalated in 1.4 turns" derived from the existing escalation
  scorer applied to the running transcript.

Data: existing `frustration.py` + transcript from `/voice/turn` +
audit log for "this turn was escalated".

### 4.5 Quality

Same data as today's tab, restyled:

- Headline ring (pass rate, with target line at 95%)
- Outcome donut (booked / info / cancelled / escalated / unresolved)
- Per-difficulty stack bar (clear / mild / medium / hard /
  emergency)
- Trend strip across last 5 evaluation runs

Data: `EvaluationReport.metrics` + `outcome_distribution` +
`by_difficulty`.

### 4.6 Escalations

- Reason histogram (already exists; restyled with weighted-score
  overlay)
- Escalated-scenarios timeline (chrono ribbon)
- Per-tenant comparison (this customer vs. all-customers
  baseline)

Data: `EvaluationReport.escalation_reason_frequency` +
`escalated_scenario_ids`.

### 4.7 Trace Explorer (drill-down)

- Span flame graph for any conversation
  (`gr.Plot` Plotly bar with start_at + duration_ms)
- Side panel: per-span attributes (cost USD, tokens, tool name,
  error)
- Replay button: re-run the same persona prompt against the same
  customer (calls `/chat` from the UI)

Data: `TraceReport.entries` + raw `traces.jsonl`.

### 4.8 Healthcare Ops (new — domain intelligence)

- Provider availability heat map (rows = providers, cols = 14
  days, cell color = % booked). From the SQLite structured store.
- No-show risk distribution (histogram of `p_no_show` for the
  next 7 days). From M3 — `NoShowPrediction` artifacts.
- PMS task queue (table). From M1 — `summary.json` + `task.json`
  on disk.
- Eligibility coverage donut. From SQLite eligibility table.

Data: existing M1 + M3 artifacts + SQLite store. No new I/O paths.

### 4.9 Cost & SLO (executive bottom strip)

- Cost ledger — rolling 24h cost per tenant (sparkline) + month
  projection
- Per-route p50/p95 latency (mini chart per route)
- Rate-limiter hits + circuit-breaker state
- Error rate per route

Data: existing structured logs + cost spans. We add a small
in-memory rolling-window aggregator at the FastAPI process level
(this is the only new server-side code — pure aggregation, no
new endpoints).

---

## 5. Reusable architecture

```
gradio_app/
├── theme.py                # custom Gradio Theme
├── style.css               # tokens + primitives + layouts
├── components.py           # pure render helpers (gr.HTML builders)
├── data_sources.py         # one façade over reports, audit, traces, SQLite
├── views/
│   ├── mission_control.py
│   ├── sentinel_ops.py     # ★
│   ├── agent_flow.py       # ★
│   ├── voice_intel.py      # ★
│   ├── quality.py
│   ├── escalations.py
│   ├── trace_explorer.py
│   ├── healthcare_ops.py
│   └── cost_slo.py
├── shell.py                # top strip + left nav + tab fan-out
├── app.py                  # thin entrypoint that composes shell + views
```

`tab_*.py` files retire after their content is split into the new
`views/` modules. The existing chat/voice clients
(`agent_client.py`, `voice_client.py`) carry forward unchanged.

Every view module exports:
- `build(state, refresh_signal) -> ViewHandle` (constructor)
- `render(customer_id, artifacts) -> tuple[Any, ...]` (data binder)

`shell.py::refresh_global` calls every view's `render()` on
customer change, exactly like the current `refresh_all`.

---

## 6. Data plumbing

We add **one** new file: `gradio_app/data_sources.py`. It wraps:

- `reports/v1.0.0/<customer>.evaluation_report.json` (locked
  schema)
- `data/<customer>/trace_<customer>.json`
- `data/<customer>/audit_<customer>.jsonl`
- `data/<customer>/pms_writeback/` (M1 artifacts)
- `data/<customer>/no_show_prediction/predictions.jsonl` (M3
  artifacts)
- `data/<customer>/structured.sqlite3` (slots + providers +
  eligibility)

Each call returns a typed dataclass. The locked schema is the
contract — `EvaluationReport`, `TraceReport`, etc., are imported
directly from `clarion.schemas`. No re-implementation; the UI
reads the same Pydantic shapes the engine writes.

No new POST endpoints. No business logic in the UI layer.

---

## 7. Build order (phased, ship-as-you-go)

Each phase is independently shippable; the existing 5 tabs keep
working until the new shell replaces them in Phase E.

| Phase | Scope | Risk |
|---|---|---|
| **A — Visual system** | `theme.py`, `style.css`, `components.py`, snapshot tests | low — purely additive |
| **B — Mission Control + Cost/SLO** | the 5-second recruiter view + bottom strip | low — read-only, new tab |
| **C — Sentinel Ops Center (★)** | the primary hero; replaces the existing Escalations content (kept as a sub-section) | medium — large visual surface |
| **D — Agent Flow (★)** | the live SVG diagram + trace integration | medium — needs span streaming from the API |
| **E — Voice Intelligence (★)** | waveform + emotion + frustration; replaces current Voice tab | medium — adds spectrogram render |
| **F — Healthcare Ops** | provider heat map + no-show + PMS queue + eligibility | low — read-only domain |
| **G — Shell rewrite** | new top strip + left nav; retire old `tab_*.py` files | medium — last phase, biggest visual change |

Each phase produces:
- A list of small atomic commits to `main`
- A green pytest run (we add snapshot + render tests per view)
- A single push to the HF Space + a build verification

Phases A–F land **without changing the existing UI** — they ship
as additional tabs alongside the current five. Phase G is the
"flip the switch" commit that moves the new shell to primary.

---

## 8. CSS strategy in one paragraph

One stylesheet (`gradio_app/style.css`) loaded via
`gr.Blocks(css=...)`. CSS variables for every color, spacing, and
type scale token. Class-based primitives (`.kpi-tile`,
`.status-badge`, `.gauge`, `.tenant-card`). Utility classes for
the mission-control grid. **No selector targeting of Gradio's
internal classes** — we style inside our own `gr.HTML` blocks
only, so a Gradio version bump can't break our visual system.
Dark mode by default; a light-mode override toggles a `.theme-light`
class on the root.

---

## 9. Deployment impact

Per-phase deploy is one Docker rebuild (~2 min) on HF Spaces.
Image-size impact is bounded:

- `theme.py` + `style.css` + `components.py`: < 50 KB total
- New views: pure Python, no model weights, no new deps
- Possible new transitive: `scipy.signal.spectrogram` is in scipy
  (already a numpy peer). NLTK VADER lexicon: ~1 MB on disk.

We continue to use the existing `scripts/serve_all.sh` single-process
container. No infra changes. No new env vars. No new secrets.

---

## 10. Definition of done for the v2 experience

A recruiter who opens
`https://huggingface.co/spaces/Ranjithmaddirala/clarion` for 30
seconds must be able to answer all five questions:

1. **What is this?** — Mission Control title + tagline answers it.
2. **Is it real?** — KPI tiles + the running cost chip + a green
   "Running" badge prove it.
3. **What does it govern?** — The Sentinel tile and Trust Gauge
   answer it.
4. **How does it work?** — The Agent Flow diagram (visible from
   the left rail) answers it.
5. **Could I work with this engineer?** — The visual polish, the
   honest gap list in the security review, the locked-schema
   evaluation report, the live voice round-trip — these answer
   it.

If all five are answerable in 30 seconds, the build is done.

---

## 11. Out of scope (explicit, so we don't drift)

- Switching UI frameworks
- Adding new POST endpoints
- Modifying any locked schema
- Adding a database
- Adding a queue / worker / cron infrastructure
- Replacing Sentinel or LangGraph
- Adding new ML models (we visualize what we already compute)
- Touching the deployment pipeline beyond the existing
  `scripts/serve_all.sh` + Docker layers

---

## 12. Open questions for confirmation before Phase A starts

These are the only things blocking implementation. Brief answers
are enough.

1. **Default mode.** Dark default with light toggle, agreed?
2. **Accent color.** Cyan `#06B6D4` for primary, or do you want a
   different brand color?
3. **Landing view.** Mission Control as the default landing page
   (replacing today's "Live Agent opens first"), agreed?
4. **Left-nav vs. top-tab.** Left-rail nav (proposed) vs. keep
   top-tabs (Gradio default). Left-rail looks more Datadog; top-tab
   stays closer to the v1 UI.
5. **Build order.** A → B → C → D → E → F → G as listed, or
   reshuffle?

Once these are answered, Phase A starts.
