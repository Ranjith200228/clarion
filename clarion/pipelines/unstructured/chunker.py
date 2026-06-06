"""Read a customer's rules corpus and split it into retrieval chunks.

Strategy
--------

The corpus is a directory of markdown files. We split on H2/H3 headings,
each becoming one chunk with the header text preserved as ``heading``
metadata. Larger sections (> ``max_chars``) are further split by paragraph
so no chunk gets so long the LLM has to skim within it.

Each chunk knows:

* ``text``     — what goes into the embedding + what the agent reads
* ``source``   — relative path to the markdown file (for citation)
* ``heading``  — last-seen heading trail (e.g. "Appointment Types > Cataract")
* ``chunk_id`` — stable id, ``<source>#<index>``

This file knows nothing about embeddings or FAISS — that's the next two
commits' job. Keeping the chunker pure makes it trivially testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# H1 = document title (kept as heading-0 context), H2/H3 = chunk boundaries.
_HEADING_RE = re.compile(r"^(?P<hashes>#{1,3})\s+(?P<text>.+?)\s*$")
_DEFAULT_MAX_CHARS = 1200


@dataclass(frozen=True)
class RuleChunk:
    chunk_id: str
    text: str
    source: str  # relative path to the source file
    heading: str  # heading trail, e.g. "Appointment Types > Cataract"


def chunk_markdown(
    text: str,
    *,
    source: str,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> list[RuleChunk]:
    """Split one markdown document into chunks."""
    lines = text.splitlines()
    # heading_trail[level] = text; H1=index 1, H2=index 2, H3=index 3
    heading_trail: list[str] = ["", "", "", ""]
    current_lines: list[str] = []
    chunks: list[RuleChunk] = []
    next_index = 0

    def _flush() -> None:
        nonlocal next_index
        body = "\n".join(current_lines).strip()
        if not body:
            return
        heading = " > ".join(h for h in heading_trail[1:] if h) or "(no heading)"
        for piece in _split_long(body, max_chars=max_chars):
            chunks.append(
                RuleChunk(
                    chunk_id=f"{source}#{next_index}",
                    text=piece,
                    source=source,
                    heading=heading,
                )
            )
            next_index += 1
        current_lines.clear()

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            level = len(m.group("hashes"))
            # H2/H3 boundary closes the current chunk.
            if level >= 2:
                _flush()
            # Update trail; H1 stays as document-level context, H2/H3 reset
            # deeper levels.
            heading_trail[level] = m.group("text").strip()
            for deeper in range(level + 1, 4):
                heading_trail[deeper] = ""
            # H1 doesn't itself produce a chunk boundary; we still want the
            # heading line in the body so the chunk has a label.
            current_lines.append(line)
            continue
        current_lines.append(line)

    _flush()
    return chunks


def chunk_rules_dir(
    rules_dir: Path,
    *,
    max_chars: int = _DEFAULT_MAX_CHARS,
) -> list[RuleChunk]:
    """Recursively chunk every ``*.md`` file under ``rules_dir``.

    Returns chunks in deterministic sorted-path order so the FAISS index
    layout is reproducible across runs.
    """
    if not rules_dir.is_dir():
        raise FileNotFoundError(f"Rules directory not found: {rules_dir}")
    out: list[RuleChunk] = []
    for md in sorted(rules_dir.rglob("*.md")):
        rel = md.relative_to(rules_dir).as_posix()
        body = md.read_text(encoding="utf-8")
        out.extend(chunk_markdown(body, source=rel, max_chars=max_chars))
    return out


def _split_long(body: str, *, max_chars: int) -> list[str]:
    """Split an over-long chunk by paragraph, never mid-sentence."""
    if len(body) <= max_chars:
        return [body]
    paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    out: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in paras:
        # If a single paragraph is already too long, accept it as-is rather
        # than cutting mid-sentence — clarity beats strict size limits.
        if len(p) > max_chars and not buf:
            out.append(p)
            continue
        if buf_len + len(p) + 2 > max_chars and buf:
            out.append("\n\n".join(buf))
            buf = [p]
            buf_len = len(p)
        else:
            buf.append(p)
            buf_len += len(p) + 2
    if buf:
        out.append("\n\n".join(buf))
    return out
