"""Runtime settings: thresholds and limits. Providers, models and prices live in utils/providers.py."""
from __future__ import annotations


MAX_TEXT_CHARS = 4000          # longer messages are truncated before the LLM sees them
AUTO_CONFIDENCE_THRESHOLD = 0.75
