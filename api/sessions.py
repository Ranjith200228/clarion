"""In-process session + per-customer cache.

The API service shares two kinds of state:

* **Per-customer resources** — CustomerConfig, StructuredStore, Retriever,
  AuditLog, TraceWriter. Loaded lazily on first request, cached for the
  lifetime of the process. Lookups are O(1) after warmup.
* **Per-conversation Agent** — one ``Agent`` per ``(customer_id, conversation_id)``
  so the rolling transcript survives between requests. Bounded by a simple
  LRU so a chatty customer can't OOM the process.

This is intentionally an in-process cache, not Redis. For MVP scope
(Phase 14 deploys to one Cloud Run instance) it's enough. Replacing
``SessionManager`` with a Redis-backed implementation is a one-file
change.
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from clarion.agents.agent import Agent
from clarion.agents.llm import LLMClient
from clarion.agents.openai_client import OpenAIClient
from clarion.config import CustomerConfig, Settings, load_customer
from clarion.observability import TraceWriter, new_trace_id
from clarion.pipelines.structured import StructuredStore
from clarion.rag.builder import load_customer_retriever
from clarion.rag.retriever import Retriever
from clarion.sentinel import AuditLog
from clarion.tools.base import ToolContext

log = logging.getLogger(__name__)


def new_conversation_id() -> str:
    """API-level conversation id. Matches the AuditLog format."""
    cid = new_trace_id().replace("trace_", "conv_", 1)
    return cid


@dataclass
class CustomerResources:
    """Per-customer artifacts that are expensive to build."""

    config: CustomerConfig
    store: StructuredStore
    retriever: Retriever | None
    audit: AuditLog
    traces: TraceWriter


@dataclass
class SessionManager:
    """Bounded session cache + per-customer resource cache.

    Thread-safe: the FastAPI default thread pool will hand requests to
    different workers, so all reads + writes hold the lock.
    """

    settings: Settings
    llm_factory: Callable[[], LLMClient]
    max_sessions: int = 256

    _customers: dict[str, CustomerResources] = field(default_factory=dict)
    _sessions: OrderedDict[tuple[str, str], Agent] = field(default_factory=OrderedDict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    # ---------- customer resources ----------

    def get_customer(self, customer_id: str) -> CustomerResources:
        with self._lock:
            cached = self._customers.get(customer_id)
            if cached is not None:
                return cached
            cached = self._build_customer(customer_id)
            self._customers[customer_id] = cached
            return cached

    def loaded_customer_ids(self) -> list[str]:
        with self._lock:
            return sorted(self._customers.keys())

    def _build_customer(self, customer_id: str) -> CustomerResources:
        log.info("loading customer %r resources", customer_id)
        cfg = load_customer(customer_id, settings=self.settings)
        store = StructuredStore.for_customer(cfg.customer_id, self.settings.data_dir)
        retriever: Retriever | None
        try:
            retriever = load_customer_retriever(cfg, data_dir=self.settings.data_dir)
        except FileNotFoundError:
            log.warning(
                "no prebuilt RAG index for %r — chat will run without retrieval",
                customer_id,
            )
            retriever = None
        audit = AuditLog.for_customer(cfg.customer_id, self.settings.data_dir)
        traces = TraceWriter.for_customer(cfg.customer_id, self.settings.data_dir)
        return CustomerResources(
            config=cfg, store=store, retriever=retriever, audit=audit, traces=traces
        )

    # ---------- session lifecycle ----------

    def get_or_create_session(
        self, customer_id: str, conversation_id: str | None
    ) -> tuple[str, Agent]:
        """Return ``(conversation_id, Agent)``. Creates the session if it
        doesn't exist; allocates a new conversation_id if the caller didn't
        provide one."""
        if conversation_id is None:
            conversation_id = new_conversation_id()
        key = (customer_id, conversation_id)
        with self._lock:
            existing = self._sessions.get(key)
            if existing is not None:
                # LRU: move to end (most recently used).
                self._sessions.move_to_end(key)
                return conversation_id, existing
            # Build a new Agent for this conversation.
            resources = self._customers.get(customer_id) or self._build_customer_locked(customer_id)
            agent = self._build_agent(resources)
            self._sessions[key] = agent
            self._evict_if_full()
            return conversation_id, agent

    def _build_customer_locked(self, customer_id: str) -> CustomerResources:
        # Called from within self._lock to avoid duplicate work.
        cached = self._customers.get(customer_id)
        if cached is not None:
            return cached
        cached = self._build_customer(customer_id)
        self._customers[customer_id] = cached
        return cached

    def _build_agent(self, resources: CustomerResources) -> Agent:
        ctx = ToolContext(customer=resources.config, structured=resources.store)
        agent = Agent(
            customer=resources.config,
            llm=self.llm_factory(),
            ctx=ctx,
            retriever=resources.retriever,
        )
        # Audit + traces use the shared per-customer writers so all
        # conversations for one customer land in the same files.
        agent.audit = resources.audit
        agent.traces = resources.traces
        return agent

    def _evict_if_full(self) -> None:
        # Cheapest-possible LRU eviction.
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)


# ---------- default LLM factory used by app.py ----------


def default_llm_factory() -> LLMClient:
    """Build the production LLM client.

    Raises if OPENAI_API_KEY is missing — tests should pass a fake
    factory via SessionManager(llm_factory=...) instead.
    """
    return OpenAIClient()


def make_session_manager(
    settings: Settings,
    *,
    llm_factory: Callable[[], LLMClient] = default_llm_factory,
    max_sessions: int = 256,
) -> SessionManager:
    return SessionManager(settings=settings, llm_factory=llm_factory, max_sessions=max_sessions)


def warmup_data_dir(path: Path) -> None:
    """Best-effort create the data dir so per-customer subdirs work
    without surprising FileNotFound on first write."""
    path.mkdir(parents=True, exist_ok=True)
