# Discovery — Cataract-Express Clinics

**Audience:** internal Clarion onboarding notes for a prospective customer
**Status:** draft, FDE-led discovery pass · 2026-06-10
**Author:** Forward Deployed Engineering

This is a sample customer-discovery doc demonstrating how Clarion is
stood up for a new vertical. It accompanies the platform README; the
demo project ships with two synthetic customers (ophthalmology,
orthopedics) and this discovery is the kind of conversation we would
run before adding a third.

---

## 1. Customer context

**Cataract-Express Clinics** runs five high-volume cataract surgery
centers across the Southeast. The customer rep we spoke with is the
Director of Patient Access; he owns three KPIs:

- **Booked-volume per call** — every front-desk call ideally results
  in a scheduled visit or a scheduled callback.
- **First-call resolution** — patients dislike being put on hold or
  promised a call back; the team wants to handle the booking inline.
- **Front-desk staff retention** — repetitive "what are your hours"
  calls drive attrition.

**Today**: three FTE phone operators per center, average ~140 calls/day
per center. Manual EMR + scheduling system, no IVR, no agent assist.

---

## 2. Discovery questions asked

The questions clustered around the four levers Clarion exposes via
`CustomerConfig`:

### Specialties + appointment types

- What are the **distinct appointment types** patients book?
- Which types are **new-patient only** vs **established only**?
- Which require **prerequisites** (dilation, imaging, NPO fast)?
- **Duration** for each type — drives slot search.

### Tools (which slots of the registry the agent can call)

- Can the agent **cancel** appointments unilaterally, or must
  cancellations route to a human (the orthopedics-style pattern)?
- Can the agent **check eligibility** in real time? — yes, they have
  a real-time Availity feed.
- Can the agent **create PMS tasks** for the front desk? — yes.

### Escalation thresholds

- What confidence floor before handoff? — the rep was comfortable
  with `0.6` (default).
- How many clarification rounds before handoff? — `3` (default).
- How is frustration handled today? — front-desk has a "warm
  handoff" SOP; we'd want to mirror that.

### Languages

- English only at intake; **Spanish** is a stretch goal for the
  Florida + Atlanta centers (~30% Spanish-preferring callers).

### Rules corpus

- They sent over a 12-page **patient access SOP**, a 4-page **NPO /
  drop / dilation checklist**, and a payer-acceptance matrix. All
  pre-existing markdown; folds straight into `data/rules/<customer>/`.

---

## 3. What we learned

1. **Cancellation is a hot button.** Their compliance team prefers
   cancellations route to a human for documentation reasons. → we'd
   ship `enabled_tools` WITHOUT `cancel_appointment` (the
   orthopedics-style configuration). Cancellation requests file a PMS
   task and tell the patient a teammate will follow up.

2. **Dilation prep is the dominant rule.** Cataract pre-op consults
   always require dilation; the rule about transport (someone must
   drive the patient home) is the most-asked question their front desk
   gets. → high-relevance chunk in the RAG corpus; the agent should be
   able to answer "do I need a ride home" without any tool call.

3. **Spanish is in scope.** Adding Spanish to `languages` in YAML is
   free; the personas + rule corpora will need translated chunks for
   high-fidelity retrieval. → Phase-2 follow-up; first launch is
   English-only.

4. **Insurance verification is a high-value automation.** The
   `check_eligibility` tool plus our payer rules corpus covers ~80%
   of "do you take X" calls. → make sure the rule chunks list every
   accepted payer explicitly so the FAQ path resolves without a tool
   call.

5. **No clinical advice** is the same hard rule everywhere. Our
   guardrails ship this by default; no per-customer change needed.

---

## 4. Recommended platform configuration

```yaml
# configs/cataract_express.yaml  (sketch — not in the repo yet)
customer_id: cataract_express
display_name: Cataract-Express Clinics
specialties:
  - Cataract Pre-Op Consult
  - Cataract Post-Op Day 1
  - Cataract Post-Op Week 1
  - Routine Eye Exam
  - Glaucoma Follow-Up
enabled_tools:
  - search_slots
  - book_appointment
  - check_eligibility
  - create_pms_task        # cancellations route here per their SOP
languages: [en]            # Spanish in phase 2
escalation:
  low_confidence: 0.6
  max_clarifications: 3
  frustration: 0.7
  on_rule_conflict: true
rules_path: rules/cataract_express
agent_persona: |
  You are Clarion, the virtual front-desk assistant for
  Cataract-Express Clinics. Warm but efficient; the patient may be
  elderly and uneasy about surgery. Always confirm whether someone can
  drive them home from a dilated visit.
```

**Onboarding time estimate**: ~1 working day. YAML + rule corpus +
synthetic personas regenerated via the simulator CLI + one eval run.

---

## 5. Risk register

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| Their EMR feed includes free-text patient notes — PHI risk in audit | Medium | High | PHI redaction (Phase 6) covers phone / email / member ids; we extend the regex pack as we encounter their format. Audit log review SOP with their privacy office. |
| Spanish at launch creates an evaluation gap | Low | Medium | Defer Spanish to phase 2 with explicit success criteria; English-only at launch. |
| Surgical urgency: "I'm losing my sight" should not be booked as routine | High | Critical | Phase 6 emergency guardrail already catches sudden vision loss / chemical splash / trauma. Audit log demonstrates 100% catch on the simulation harness. |
| Dilation rule retrieval misses on common phrasings | Medium | Medium | Add explicit phrasings to RAG corpus; the evaluation harness's `safety_catch_rate` and per-intent breakdown surface misses. |
| Front desk perceives the agent as a threat to jobs | Medium | High | Position as "lifts the repetitive 30% so you focus on the urgent 70%"; their KPI is staff retention, which our escalation rate directly serves. |

---

## 6. Success metrics (first 90 days)

| Metric | Source | Target |
|---|---|---|
| Containment rate on FAQ + booking | `report_cataract_express.json` | >= 60% |
| Booking accuracy | `report` | >= 95% |
| Hallucination rate (judge) | `report` | < 5% |
| Safety catch rate | `report` | 100% |
| Escalation recall | `report` | >= 90% |
| Cost per request | `report` | < $0.01 |
| Front-desk hours displaced (estimated) | their internal time tracking | >= 1.5 FTE-hour/day per center |

---

## 7. Open questions for the next sync

- Real-time eligibility latency budget — what's the timeout we should
  apply on the Availity call?
- Are they using Athena, Modmed, or NextGen? — drives the PMS task
  export pipeline (post-launch module M1).
- Audit log retention — what's their compliance team's preference?

---

## 8. Decision

**Recommend proceeding** with Cataract-Express as the third demo
customer. The configuration is mostly a YAML + rules-corpus delta on
the existing ophthalmology customer (they share appointment types and
the dilation rule story). Cancellation routing follows the orthopedics
pattern. ~1 day to stand up; ~3 days to ship with an eval-backed launch
readout.
