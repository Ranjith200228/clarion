"""Post-launch optional modules.

Each module is opt-in per customer via ``CustomerConfig.modules`` and
lives in its own subpackage:

  pms_writeback/    — M1: summary.json + task.json per conversation
  (future)
  no_show/          — M3: XGBoost no-show probability
  voice/            — M5: STT + TTS shell around the text core

Modules import from ``clarion.schemas`` and read completed transcripts.
They never modify agent code. The spec rule "Modules must remain
isolated. No hard coupling." is enforced by this dependency direction.
"""
