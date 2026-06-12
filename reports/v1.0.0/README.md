# Clarion v1.0.0 — final evaluation reports

These are the locked-schema `EvaluationReport` artifacts produced
by the scripted harness against the v1.0.0 release. Two customers
shipped (`ophthalmology`, `orthopedics`), each scored across the
full 100-scenario corpus.

| Customer | Scenarios | Pass rate | Containment | Booking | Halluc. | Esc. P / R | Safety |
|---|---|---|---|---|---|---|---|
| ophthalmology | 100 | 100% | 74.0% | 100% | 0.0% | 0.62 / 1.00 | 100% |
| orthopedics   | 100 | 100% | 66.0% | 100% | 0.0% | 0.79 / 1.00 | 100% |

* **Pass rate** — fraction of scenarios where the actual outcome
  matched ground truth.
* **Containment** — fraction resolved without a human handoff.
* **Booking accuracy** — for scenarios whose expected outcome is
  `booked`, the fraction that actually booked correctly.
* **Hallucination rate** — average judge.hallucination across
  scenarios that had a judge attached.
* **Escalation precision / recall** — against
  `ground_truth.should_escalate`.
* **Safety catch rate** — recall on emergency + clinical-advice
  intents.

Reports were generated via:

```bash
poetry run python -m clarion.evaluation.cli run all --out reports/v1.0.0
```

They conform to `EvaluationReport` schema version `1.0.0`
(`REPORT_SCHEMA_VERSION` constant in
[`clarion/schemas/evaluation.py`](../../clarion/schemas/evaluation.py)).
The Phase 14 Gradio UI re-reads these files directly — no
recomputation happens in the dashboard.

## Notes

* Booking accuracy and safety catch rate both hit 100% — the agent
  books correctly on every scenario whose ground truth said it
  should, and never lets an emergency or clinical-advice prompt
  reach the model.
* Hallucination rate is 0% on this corpus because every
  scripted-mode test runs against `FakeLLM`. Live-mode numbers
  with a real OpenAI backend live in the Gradio app's Evaluation
  tab.
* Escalation precision differs across customers because
  orthopedics has more ambiguous (medium-difficulty) scenarios,
  which means the scorer fires fewer false positives. Recall is
  100% on both — no real escalation gets missed.
