# Load testing

Two surfaces here serve different purposes:

* **`locustfile.py`** — interactive, distributed-capable load test
  that drives a *running* Clarion instance over HTTP. Run it before
  cutting a release against a staging deployment or a local
  uvicorn process.
* **`in_process_load.py`** — fast, in-process burst test that runs
  against a FastAPI `TestClient` with a `FakeLLM` wired in. Used
  by `tests/loadtest/test_p95_sla.py` to enforce the SLA in CI
  without needing a real OpenAI key.

## Service-level objective

| Endpoint | Metric | Target |
|---|---|---|
| `POST /chat` (FakeLLM) | p50 latency | < 200 ms |
| `POST /chat` (FakeLLM) | p95 latency | < 500 ms |
| `POST /chat` (FakeLLM) | error rate | < 0.5% |

These are the **in-process** numbers — they isolate the agent loop +
tool dispatch + middleware overhead from network and model latency.
For end-to-end SLOs against a real OpenAI backend, expect p95 to
climb into 2-3 s territory (a typical `gpt-4o-mini` call dominates).

## Running locust against a live server

```bash
# Terminal 1: start the API
poetry run python -m api.app --port 8080

# Terminal 2: locust
poetry run pip install locust
poetry run locust -f loadtest/locustfile.py --host http://localhost:8080
```

Then open <http://localhost:8089>, dial in 50 users / 5 spawn-rate,
and watch the median + p95 columns.

## Running the in-process burst

```bash
poetry run pytest tests/loadtest/ -m loadtest
```

The `loadtest` marker is off by default in CI; it runs locally on
demand and on release branches via the `release` workflow.
