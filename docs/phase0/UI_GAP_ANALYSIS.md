# UI_GAP_ANALYSIS.md — what makes Clarion look like a prototype today

*Phase 0 deliverable. Each gap is concrete (named, locatable in code,
verifiable by a recruiter) and tied to the five pillars from the
vision brief.*

The honest summary at the top: **the engine is staff-level, the
visualization is intern-level.** The agent, trust engine, evaluation,
multi-tenant config, observability — all are doing the right things.
But the surface a recruiter loads in 5 seconds shows a default Gradio
theme with 5 utilitarian tabs and no narrative. That gap is what
this document inventories.

---

## 1. What currently looks like a prototype

### 1.1 Chrome & framing
- **No top-level brand bar** beyond a single `gr.Markdown("# Clarion …")`
  line — no logo, no version, no environment badge, no SLO banner.
- **No global navigation** — the customer dropdown is the only
  cross-tab control, and it's a stock `gr.Dropdown` styled with the
  default Gradio theme.
- **Default Gradio theme** end-to-end — light mode, the standard
  rounded chrome, no custom palette. Every Gradio Space looks like
  this; nothing telegraphs "platform."
- **No iconography** anywhere — pillar names, agent identities, and
  trust signals are surfaced as plain text.

### 1.2 Live Agent tab
- `gr.ChatInterface` is the most common Gradio idiom on the web; a
  recruiter has seen it 100 times this week.
- "Running cost: $0.000000 · tokens in/out: 0/0" is a plain markdown
  line — there is no visual weight on the cost story, no spark line,
  no per-turn breakdown.
- The trust engine that runs every turn (guardrails → judge →
  escalation scorer) is **completely invisible** in the UI. The user
  sees a reply; they do not see the four checks that produced it.
- The tool-call timeline that the trace already captures is **not
  rendered inline** — you have to open the Trace Explorer tab to
  realize tools were called at all.

### 1.3 Voice Agent tab
- The microphone widget and audio playback work, but the tab is just
  a `gr.Audio` in → `gr.Audio` out. There is **no waveform**, **no
  spectrogram**, **no emotion meter**, **no frustration trace**.
- Per-stage latency is shown as a markdown one-liner. Tesla/Datadog
  would show three sparkline gauges with green/amber/red thresholds.

### 1.4 Quality / Escalations / Trace Explorer tabs
- All three are **gr.Dataframe + gr.Markdown**. The reports underneath
  are rich (six headline numbers, outcome distribution, reason
  histograms, escalated scenario ids, per-trace spans with cost) but
  the rendering treats them as spreadsheets.
- **No visual hierarchy.** Every number is the same size, weight, and
  color as every other number. A recruiter scanning for "does this
  thing work" cannot find the answer in three seconds.
- **No comparison.** Switching customer reloads the panel but does
  not show a delta — "ophthalmology contains 6 points more than
  orthopedics on safety_catch_rate" is information we already have
  and don't surface.

### 1.5 No mission-control landing
- The app opens directly into Live Agent. A recruiter never sees a
  summary view. There is no "Operations" home page showing **all
  customers live**, current SLA, current emergency rate, current
  escalation rate, current cost burn, recent incidents. This is the
  single biggest gap.

### 1.6 Trust engine: present in code, absent in UI
- The Sentinel package (`clarion/sentinel/`) is the strongest piece
  of engineering in the repo. It computes a composite escalation
  score from five weighted signals, runs an LLM judge, redacts PHI,
  appends to an audit log, and short-circuits on emergency phrases.
- **None of that is visible.** The user sees `escalation_score:
  0.62` as a number on a chat reply. No gauge. No timeline. No
  reason breakdown. No PHI redaction badge. No "judge said: 0.81
  confident, 0 hallucinations" indicator.
- The vision brief explicitly calls Sentinel the **primary hero
  feature**. Today it has zero dedicated surface.

### 1.7 Multi-agent reasoning: invisible
- The LangGraph backend is opt-in via YAML and works correctly, but
  the UI **does not show the graph**. A user typing "I need a
  cataract pre-op" has no idea their message was classified by a
  router, dispatched to `BookingSpecialist`, hit a search_slots
  tool, returned through the supervisor, and finished.
- The five specialists, their tool subsets, their per-call personas
  — none of this is rendered. The graph is the second hero feature
  per the vision brief and today it is purely backend.

### 1.8 Observability: present in code, absent in the dashboard
- We compute per-call cost, per-stage latency, tokens in/out,
  correlation IDs, full span trees. **The dashboard surfaces almost
  none of this in an operational view.**
- There is no real-time stream of incoming requests, no rolling
  p50/p95 graph, no error rate per route, no rate-limit hit counter,
  no circuit breaker state indicator.

### 1.9 Healthcare-domain intelligence: absent
- The product is positioned as healthcare operations, but nothing on
  screen feels healthcare-shaped. There are no:
  - Provider schedules
  - Slot availability heat map
  - Patient eligibility risk distribution
  - No-show risk overlay (we ship M3 that computes this — it is not
    surfaced anywhere in the UI)
  - PMS task queue (M1 ships the writeback, but the
    `summary.json` + `task.json` files land on disk and disappear
    from the user's view)

### 1.10 Executive view: absent
- The Definition-of-Done table at the top of the GitHub README is
  the closest thing to an executive view, but it lives in
  Markdown. The application itself has no place where the recruiter
  can see, in one glance:
  - **How many customers** are running
  - **What % of calls** are contained
  - **How many emergencies caught**
  - **$ saved vs. front-desk baseline** (or projected)
  - **Trust score** (a single visible 0-100 number)

---

## 2. What should become enterprise-grade

A single-sentence answer per surface:

| Surface | Today | Enterprise-grade |
|---|---|---|
| **Landing** | Live Agent tab opens by default | Mission Control home with global KPIs + per-tenant health + recent incidents |
| **Live Agent** | Plain chat | Chat **alongside** a live Sentinel score gauge + tool-timeline + judge verdict + trust signal stack |
| **Voice Agent** | Mic in, audio out | Waveform + emotion meter + frustration trace + transcript with confidence shading + per-stage latency rings |
| **Trust** | (no surface) | Dedicated **Sentinel Operations Center** view — gauge, signal breakdown, reasons histogram, recent escalations stream, PHI redaction counter, judge confidence trend |
| **Multi-agent** | (no surface) | Live **Agent Flow** visualization — animated graph showing router → specialist → supervisor for the current turn, with each node lit when active and the tool calls listed in-line |
| **Quality** | 2 dataframes | Headline ring (pass rate, containment, safety) + outcome donut + per-difficulty stack bars |
| **Escalations** | 2 dataframes | Reason histogram with weighted-score overlay + escalated scenarios timeline + per-tenant comparison |
| **Trace Explorer** | Tall dataframe | Span flame graph + per-turn drilldown panel + cost+latency sparkline strip |
| **Healthcare intelligence** | (no surface) | Slot availability heat map + no-show risk distribution + PMS task queue |
| **Executive view** | (no surface) | Single-page printout: 8 KPIs + tenant table + cost ledger + SLO trail |

---

## 3. Missing operational views (named)

These are the views that exist in a Datadog / Tesla mission-control
console and don't exist here:

1. **Mission Control** — the home page. Header strip of 8 KPI tiles,
   tenant table with health-score column, recent emergency stream,
   recent escalation stream.
2. **Sentinel Operations Center** — the hero. Composite trust score
   gauge (live), signal-by-signal breakdown bar, scorer-vs-judge
   delta over time, PHI redaction count, audit-log tail.
3. **Agent Flow** — the animated graph from the vision brief.
   `Patient request → Router → Specialist → Retrieval → Tool →
   Sentinel → Response`. Each node carries a state badge + the time
   spent + the cost incurred.
4. **Voice Intelligence** — waveform live, transcript live (with
   per-token confidence), emotion meter, frustration trace, voice
   biomarkers, escalation prediction overlay.
5. **Cost & SLO** — rolling cost ledger, per-route p50/p95 latency,
   rate-limit hits, circuit breaker state, error rate per route.
6. **Healthcare Operations** — provider schedule, slot heat map,
   eligibility status distribution, no-show risk leaderboard, PMS
   task queue.
7. **Customer Switcher (real)** — current dropdown becomes a
   first-class side panel listing every tenant with health,
   compliance posture, last-built-at, custom badge color.
8. **Trace Drill-down** — flame graph for any conversation, span
   detail pop-out, replay button to re-run the same prompt against
   the same persona.

---

## 4. Missing executive dashboards

For a recruiter and a hiring manager (separate audiences, both have
to walk away convinced):

- **Recruiter view**: 8 KPI tiles + one-liner under each + Trust
  Score (the single number). Must be readable in 5 seconds.
- **Hiring manager view**: the operational console named above.
  Must be drill-able for 5 minutes without hitting a dead end.
- **Engineering manager view**: trace flame graph + cost ledger +
  SLO trail. Must answer "is this fast, cheap, correct?" in 30
  seconds.

---

## 5. Missing healthcare-intelligence views

- **Provider availability heat map** — calendar grid (rows =
  providers, columns = days), cells colored by % booked. Data is
  already in the SQLite store; we render zero of it.
- **No-show risk distribution** — histogram of `p_no_show` scores
  for the next 7 days, with risk bands overlaid. M3 already
  computes this; the dashboard never shows it.
- **PMS task queue** — the `summary.json` + `task.json` writeback
  artifacts M1 produces are written to disk and never displayed.
  A simple table listing tasks with `priority`, `assignee_group`,
  and a one-line summary would close the loop.
- **Eligibility coverage** — pie chart of payer distribution,
  eligibility-check pass rate, denial reasons. We capture the data;
  we don't render it.
- **Emergency catch board** — running count of guardrail-fired
  emergencies, per-tenant, with timestamps. The audit log already
  carries this.

---

## 6. The single biggest gap (one sentence)

The product looks like a prototype because **the trust engine — the
strongest piece of engineering in the repo — has no UI**. Fixing
just that one gap moves Clarion from "neat student project" to
"this person knows what enterprise AI governance looks like."

---

## 7. What we are *not* changing (out-of-scope reminder)

- Gradio is the UI framework. We will write a custom theme, custom
  CSS, custom HTML/SVG panels, and lots of `gr.Plot` — but we are
  not switching to Next.js or React.
- The API contract stays. Every new view reads from existing
  endpoints or from existing on-disk artifacts. No new POST routes
  required for v2.
- The locked report schemas stay. Every new chart consumes
  `EvaluationReport` / `TraceReport` / module-output JSON as-is.
- The agent stack stays. Sentinel, LangGraph, FAISS, Whisper, TTS,
  and the observability layer are not being rebuilt.

---

The next document (`IMPLEMENTATION_PLAN.md`) names the components
we build, the order we build them in, the CSS strategy, and the
deployment impact.
