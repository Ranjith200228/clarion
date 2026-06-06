"""End-to-end tests for the FAISS retriever.

These use the real ophthalmology and orthopedics rules corpora to enforce
Phase 3's acceptance criterion: *RAG retrieves correct rules.* Each test
asks a question whose answer lives in exactly one file, then asserts that
file appears in the top-k results — which is how the agent actually
consumes retrieval output (it grounds the LLM with k chunks, not just one).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from clarion.config import Settings, load_customer
from clarion.pipelines.unstructured import chunk_rules_dir
from clarion.rag.embeddings import TfidfEmbedder
from clarion.rag.retriever import Retriever

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = REPO_ROOT / "configs"
DATA_DIR = REPO_ROOT / "data"


def _build_retriever(rules_dir: Path, tmp_path: Path) -> Retriever:
    chunks = chunk_rules_dir(rules_dir)
    assert chunks, f"No chunks built from {rules_dir} — corpus missing?"
    return Retriever.build_and_save(chunks, embedder=TfidfEmbedder(), out_dir=tmp_path)


@pytest.fixture
def ophthalmology_retriever(tmp_path: Path) -> Retriever:
    cfg = load_customer(
        "ophthalmology",
        settings=Settings(customer="ophthalmology", config_dir=CONFIGS_DIR, data_dir=DATA_DIR),
    )
    return _build_retriever(cfg.rules_path, tmp_path)


@pytest.fixture
def orthopedics_retriever(tmp_path: Path) -> Retriever:
    cfg = load_customer(
        "orthopedics",
        settings=Settings(customer="orthopedics", config_dir=CONFIGS_DIR, data_dir=DATA_DIR),
    )
    return _build_retriever(cfg.rules_path, tmp_path)


# ---------- ophthalmology: top-1 must be in the right file ----------


@pytest.mark.parametrize(
    "query, expected_source",
    [
        ("how long is a cataract pre-op consult?", "01_appointment_types.md"),
        (
            "do I need to bring sunglasses and arrange transportation home?",
            "03_dilation_and_prep.md",
        ),
        ("do you accept kaiser insurance?", "04_insurance_and_payers.md"),
        ("what is the cancellation policy?", "05_cancellation_and_reschedule.md"),
        ("the patient suddenly lost vision in one eye", "06_emergencies_and_escalation.md"),
    ],
)
def test_ophthalmology_query_finds_correct_file_in_top_k(
    ophthalmology_retriever: Retriever, query: str, expected_source: str
) -> None:
    hits = ophthalmology_retriever.retrieve(query, k=3)
    assert hits, "retriever returned nothing"
    sources = [h.chunk.source for h in hits]
    assert expected_source in sources, (
        f"query={query!r}\n"
        f"expected {expected_source} in top-3, got {sources}\n"
        f"all hits: {[(h.chunk.source, round(h.score, 3)) for h in hits]}"
    )


# ---------- orthopedics: top-1 must be in the right file ----------


@pytest.mark.parametrize(
    "query, expected_source",
    [
        ("how long is a new patient joint consult?", "01_appointment_types.md"),
        ("workers comp claim number", "03_workers_comp.md"),
        ("do you accept medicaid?", "04_insurance_and_payers.md"),
        ("how do I cancel my appointment?", "05_cancellation_and_reschedule.md"),
        ("possible compound fracture", "06_emergencies_and_escalation.md"),
    ],
)
def test_orthopedics_query_finds_correct_file_in_top_k(
    orthopedics_retriever: Retriever, query: str, expected_source: str
) -> None:
    hits = orthopedics_retriever.retrieve(query, k=3)
    assert hits, "retriever returned nothing"
    sources = [h.chunk.source for h in hits]
    assert expected_source in sources, (
        f"query={query!r}\n"
        f"expected {expected_source} in top-3, got {sources}\n"
        f"all hits: {[(h.chunk.source, round(h.score, 3)) for h in hits]}"
    )


# ---------- cross-customer isolation ----------


def test_workers_comp_query_finds_nothing_in_ophthalmology(
    ophthalmology_retriever: Retriever,
) -> None:
    """workers' comp content lives only in orthopedics; ophthalmology should
    not surface a workers-comp-themed top hit."""
    hits = ophthalmology_retriever.retrieve("workers compensation claim", k=3)
    assert hits
    for h in hits:
        assert (
            "workers" not in h.chunk.text.lower()
        ), f"unexpected workers-comp content in ophthalmology: {h.chunk.source}"


def test_dilation_query_finds_nothing_in_orthopedics(
    orthopedics_retriever: Retriever,
) -> None:
    """Dilation is an ophthalmology-only concept."""
    hits = orthopedics_retriever.retrieve("pupil dilation drops", k=3)
    for h in hits:
        assert (
            "dilat" not in h.chunk.text.lower()
        ), f"unexpected dilation content in orthopedics: {h.chunk.source}"


# ---------- load round-trip ----------


def test_build_and_load_round_trip(tmp_path: Path) -> None:
    cfg = load_customer(
        "ophthalmology",
        settings=Settings(customer="ophthalmology", config_dir=CONFIGS_DIR, data_dir=DATA_DIR),
    )
    chunks = chunk_rules_dir(cfg.rules_path)
    built = Retriever.build_and_save(chunks, embedder=TfidfEmbedder(), out_dir=tmp_path)
    built_hits = built.retrieve("how long is a cataract consult?", k=2)

    loaded = Retriever.load(tmp_path, embedder=TfidfEmbedder())
    loaded_hits = loaded.retrieve("how long is a cataract consult?", k=2)

    assert [(h.chunk.chunk_id, round(h.score, 4)) for h in built_hits] == [
        (h.chunk.chunk_id, round(h.score, 4)) for h in loaded_hits
    ]


# ---------- defensive ----------


def test_empty_query_returns_empty(ophthalmology_retriever: Retriever) -> None:
    assert ophthalmology_retriever.retrieve("", k=3) == []
    assert ophthalmology_retriever.retrieve("   ", k=3) == []
