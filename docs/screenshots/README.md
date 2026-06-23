# Screenshots

The repo README embeds each of these PNGs. Capture them from a
running Gradio app at `http://localhost:7860/` (or the deployed
HF Space). One PNG per filename below — names are referenced
directly by `../../README.md`.

## What to capture

| Filename                     | Tab / view                      | What to show                                                                                                    |
| ---------------------------- | ------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `01-hero.png`                | Mission Control (full page)     | The whole dashboard at a glance: brand strip, greeting, *Today's Standout* card, KPI tiles, tenant table.       |
| `02-mission-control.png`     | Mission Control (KPI strip)     | Close-up of the 8 KPI tiles + sparklines, then the comparative tenant strip below.                              |
| `03-sentinel-ops.png`        | Sentinel Operations             | Trust gauge + signal panel + judge panel + audit-tail panel. Shows the per-trace verdicts surface.              |
| `04-agent-flow.png`          | Agent Flow                      | The multi-agent graph diagram: router → specialist → supervisor with the chosen path highlighted.               |
| `05-voice-intelligence.png`  | Voice Intelligence              | Emotion donut ring + frustration trace + sample transcript panel.                                               |
| `06-patient-360.png`         | Patient 360                     | Patient roster chips + profile card + appointment confirmation panel with the **Download confirmation** button. |
| `07-cost-slo-ocr.png`        | Cost & SLO (scrolled to OCR)    | The invoice OCR widget: upload + extract button + the structured "Uploaded invoices" result card.               |
| `08-live-agent.png`          | Live Agent                      | Mid-conversation chat with the LangGraph agent; tool calls + cost surfaced after a turn.                        |
| `09-voice-agent.png`         | Voice Agent                     | The voice tab in mid-conversation; record/stop controls + reply audio + per-stage latency table.                |
| `10-configuration.png`       | Configuration                   | Identity card + enabled tools chips + escalation threshold bars + agent persona panel.                          |
| `11-customer-switch.gif`     | Top of any tab during switch    | (Optional) 3-second GIF of the customer dropdown changing tenants — accent ripples through KPI edges + badges.   |

## How to capture

1. Run the app: `poetry run python -m gradio_app`, open
   `http://localhost:7860/` in Chrome or Edge.
2. Set the browser zoom to 100% for consistency.
3. Use the OS snipping tool (or `Win + Shift + S` on Windows /
   `Cmd + Shift + 4` on macOS) to capture each region.
4. Save as PNG with the exact filename above into this folder.
5. The README will pick them up automatically.

> The GIF is optional but lands the "this is alive" feeling
> better than any static frame. ScreenToGif on Windows or
> Kap on macOS work well; keep the file under 5 MB so GitHub
> renders it inline.
