"""Module M1: PMS Writeback.

Convert completed conversations into two structured artifacts per call:

  summary.json   -> ConversationSummary
  task.json      -> PmsTaskWriteback

These land at ``<data_dir>/<customer_id>/pms_writeback/<conversation_id>/``
and are the wire shape downstream PMS systems consume.

Public surface:
  HeuristicExtractor — regex-based extractor (no LLM key needed)
  Extractor          — Protocol that future LLM-backed extractors satisfy
"""

from clarion.modules.pms_writeback.extractor import (
    ExtractionContext,
    Extractor,
    HeuristicExtractor,
)

__all__ = [
    "ExtractionContext",
    "Extractor",
    "HeuristicExtractor",
]
