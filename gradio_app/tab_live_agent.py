"""Live Agent tab — gr.ChatInterface backed by the FastAPI /chat endpoint.

The spec's required-display list:
* escalation score
* last tool call
* running cost

All three come from ``ChatResponse.last_turn_metrics`` (Phase 14 commit
4 added them to the API). The UI just renders the values it receives.

The tab does not import ``clarion.agents`` — the agent runtime stays in
the FastAPI process. The Phase 15 container will run both processes
side-by-side; ``CLARION_API_URL`` selects the backend in any
environment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import gradio as gr

from gradio_app.agent_client import AgentClient, ApiError

log = logging.getLogger(__name__)


@dataclass
class LiveAgentState:
    """Mutable per-session state held in ``gr.State``."""

    customer_id: str = "ophthalmology"
    conversation_id: str | None = None
    running_cost_usd: float = 0.0
    running_input_tokens: int = 0
    running_output_tokens: int = 0


@dataclass
class LiveAgentTab:
    state: gr.State
    metrics_md: gr.Markdown
    chat_interface: gr.ChatInterface


def build(client: AgentClient | None = None) -> LiveAgentTab:
    """Build the Live Agent tab.

    Pass ``client`` for tests; production callers let the function
    construct a default ``AgentClient`` keyed on env vars.
    """
    backend = client or AgentClient()

    gr.Markdown(
        "## Live Agent\n\n"
        "Talk to Clarion. The agent's tool calls, escalation score, "
        "and running cost are surfaced below after each turn."
    )

    state = gr.State(LiveAgentState())
    metrics_md = gr.Markdown(_render_metrics(LiveAgentState()))

    def respond(  # type: ignore[no-untyped-def]
        message: str,
        history,  # gradio history value, not used directly
        st: LiveAgentState,
    ):
        """One chat turn: POST /chat, update state, return reply + metrics."""
        try:
            turn = backend.chat(
                customer_id=st.customer_id,
                message=message,
                conversation_id=st.conversation_id,
            )
        except ApiError as e:
            return (
                f"[agent backend error] {e}\n\n"
                f"_If you're running locally, start the API with:_ "
                f"`python -m api.app` _on port 8000._"
            )

        st.conversation_id = turn.conversation_id
        st.running_cost_usd += turn.cost_usd
        st.running_input_tokens += turn.input_tokens
        st.running_output_tokens += turn.output_tokens
        suffix = _render_metrics(st, last_turn=turn)
        return f"{turn.reply}\n\n---\n{suffix}"

    chat = gr.ChatInterface(
        fn=respond,
        additional_inputs=[state],
        type="messages",
        title=None,
        description=None,
    )

    return LiveAgentTab(state=state, metrics_md=metrics_md, chat_interface=chat)


def set_customer(st: LiveAgentState, customer_id: str) -> LiveAgentState:
    """Reset the running totals when the customer switcher changes."""
    return LiveAgentState(customer_id=customer_id)


# ---------- helpers ----------


def _render_metrics(state: LiveAgentState, last_turn=None) -> str:  # type: ignore[no-untyped-def]
    """Compact markdown summary of the running session state."""
    parts = [f"**Customer:** `{state.customer_id}`"]
    if state.conversation_id:
        parts.append(f"**Conversation:** `{state.conversation_id[:18]}…`")
    if last_turn is not None:
        if last_turn.escalation_score is not None:
            parts.append(f"**Last escalation score:** {last_turn.escalation_score:.2f}")
        if last_turn.last_tool_call:
            parts.append(f"**Last tool call:** `{last_turn.last_tool_call}`")
    parts.append(
        f"**Running cost:** ${state.running_cost_usd:.6f} · "
        f"tokens in/out: {state.running_input_tokens}/{state.running_output_tokens}"
    )
    return " · ".join(parts)
