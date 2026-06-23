# Screenshots

Real captures of the Clarion dashboard running at `localhost:7860`,
cropped to just the application content (no Chrome chrome, no
Windows taskbar). Each PNG is referenced from the main
[`../../README.md`](../../README.md).

## Shipped

| File | View | Note |
|---|---|---|
| `03-sentinel-ops.png` | Sentinel Operations | Trust score + sentinel score + composite trust gauge + signal breakdown. |
| `04-agent-flow.png` | Agent Flow | Live path through the multi-agent graph for `ophthalmology_clear_book_001`. |
| `05-voice-intelligence.png` | Voice Intelligence | Emotion donut, frustration trace, escalation prediction. |
| `06-patient-360.png` | Patient 360 | Patient roster + profile card with engagement/sentiment/trust scores. |
| `07-cost-slo.png` | Cost & SLO (top) | KPI strip + cost-share donut + per-tenant cost breakdown. |
| `07-cost-slo-ocr.png` | Cost & SLO (OCR) | Invoice OCR widget — upload + extract control. The OCR view the main README links to. |
| `08-live-agent.png` | Live Agent | Page intro + chatbot canvas + retry/undo/clear controls. |
| `09-voice-agent.png` | Voice Agent | Page intro + record button + microphone select + Clarion reply slot. |

## Still to capture (optional polish)

| File | View |
|---|---|
| `01-hero.png` | Mission Control with brand strip + greeting + Today's Standout + first KPI row. The single best opener for the README. |
| `02-mission-control.png` | Mission Control close-up: KPI strip + Tenants-Live tile + the Tenant-by-Tenant snapshot row. |
| `10-configuration.png` | Configuration tab: identity card + enabled-tools chips + escalation thresholds + agent persona panel. |

## How to capture

1. Run the app: `poetry run python -m gradio_app`, open
   `http://localhost:7860/` in Chrome or Edge.
2. Set the browser zoom to 100%.
3. Use the OS snipping tool (`Win + Shift + S` on Windows /
   `Cmd + Shift + 4` on macOS) and drag a rectangle around just
   the Clarion content area &mdash; skip the URL bar at top and
   the taskbar at bottom.
4. Save as PNG with the exact filename above into this folder.
5. The main README will pick it up automatically on the next push.
