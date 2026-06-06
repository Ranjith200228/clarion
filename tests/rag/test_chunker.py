"""Pure-text tests for the markdown chunker (no embeddings, no I/O)."""

from __future__ import annotations

from pathlib import Path

import pytest
from clarion.pipelines.unstructured.chunker import chunk_markdown, chunk_rules_dir


def test_h2_starts_new_chunk() -> None:
    md = "# Doc\n\n## Section A\nAlpha body.\n\n## Section B\nBeta body.\n"
    chunks = chunk_markdown(md, source="x.md")
    assert len(chunks) == 2
    assert chunks[0].heading == "Doc > Section A"
    assert "Alpha body" in chunks[0].text
    assert chunks[1].heading == "Doc > Section B"
    assert "Beta body" in chunks[1].text


def test_h3_starts_new_chunk_under_h2() -> None:
    md = "# Doc\n\n## Section\nIntro\n\n### Subsection\nDetail.\n"
    chunks = chunk_markdown(md, source="x.md")
    headings = [c.heading for c in chunks]
    assert "Doc > Section" in headings
    assert "Doc > Section > Subsection" in headings


def test_no_heading_yields_single_unheaded_chunk() -> None:
    chunks = chunk_markdown("just a body, no headings", source="x.md")
    assert len(chunks) == 1
    assert chunks[0].heading == "(no heading)"


def test_chunk_ids_are_stable_and_unique() -> None:
    md = "## A\nbody a\n\n## B\nbody b\n"
    chunks = chunk_markdown(md, source="rules.md")
    ids = [c.chunk_id for c in chunks]
    assert ids == ["rules.md#0", "rules.md#1"]
    assert len(set(ids)) == len(ids)


def test_long_section_splits_on_paragraphs() -> None:
    body = "\n\n".join([f"Paragraph {i}." * 30 for i in range(6)])
    md = f"## Big\n{body}\n"
    chunks = chunk_markdown(md, source="big.md", max_chars=400)
    assert len(chunks) > 1
    # No chunk should split mid-paragraph — every chunk ends with "."
    for c in chunks:
        assert c.text.strip().endswith(".")


def test_chunk_rules_dir_walks_recursively(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("## A\nbody a\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("## B\nbody b\n", encoding="utf-8")

    chunks = chunk_rules_dir(tmp_path)
    sources = sorted(c.source for c in chunks)
    assert sources == ["a.md", "sub/b.md"]


def test_chunk_rules_dir_missing_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        chunk_rules_dir(tmp_path / "does_not_exist")
