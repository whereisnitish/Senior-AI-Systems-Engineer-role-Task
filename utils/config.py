"""Runtime settings: model, endpoint, pricing and thresholds."""
from __future__ import annotations

import os


DEFAULT_MODEL = "qwen/qwen3.8-27b:free"
OPENROUTER_URL = os.environ.get("OPENROUTER_URL", "https://openrouter.ai/api/v1/chat/completions")
MAX_TEXT_CHARS = 4000          # longer messages are truncated before the LLM sees them
AUTO_CONFIDENCE_THRESHOLD = 0.75

# Paid-tier price of the default model on OpenRouter (USD per token), checked 2026-10-02.
# The :free variant is $0 but rate-limited (50 req/day without credits) and not usable for production volume.
PRICED_MODEL = "qwen/qwen3.8-27b"
PRICE_IN = 0.42 / 1_000_000
PRICE_OUT = 3.00 / 1_000_000
