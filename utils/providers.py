"""LLM providers the tool can use, how one is picked automatically, and per-model prices for the cost projection."""
from __future__ import annotations

import os

# protocol "openai"    -> OpenAI-compatible /chat/completions, standard library only (services/llm/openai_compatible.py)
# protocol "anthropic" -> Claude Messages API through the official SDK (services/llm/anthropic_client.py)
#
# reasoning_style says how --reasoning is sent on OpenAI-compatible endpoints:
#   "openrouter"    reasoning: {enabled: false} or {effort: ...}
#   "openai"        reasoning_effort, with "off" sent as "low" (the lowest level every OpenAI reasoning model takes)
#   "effort_if_set" reasoning_effort only when a level is chosen; nothing for "off"
PROVIDERS = {
    "openrouter": dict(protocol="openai", base_url="https://openrouter.ai/api/v1", key_env="OPENROUTER_API_KEY",
                       default_model="qwen/qwen3.8-27b:free", reasoning_style="openrouter",
                       send_temperature=True, token_param="max_tokens", headers={"X-Title": "message-triage"}),
    "openai": dict(protocol="openai", base_url="https://api.openai.com/v1", key_env="OPENAI_API_KEY",
                   default_model="gpt-6-luna", reasoning_style="openai",
                   send_temperature=False, token_param="max_completion_tokens"),
    "anthropic": dict(protocol="anthropic", base_url=None, key_env="ANTHROPIC_API_KEY",
                      default_model="claude-opus-5"),
    "groq": dict(protocol="openai", base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY",
                 default_model=None, reasoning_style="effort_if_set"),
    "gemini": dict(protocol="openai", base_url="https://generativelanguage.googleapis.com/v1beta/openai",
                   key_env="GEMINI_API_KEY", default_model=None, reasoning_style="effort_if_set"),
    "ollama": dict(protocol="openai", base_url="http://localhost:11434/v1", key_env=None, needs_key=False,
                   default_model=None, reasoning_style="effort_if_set"),
    # anything else that speaks the OpenAI chat-completions protocol (vLLM, LM Studio, Together, a proxy, ...)
    "openai-compatible": dict(protocol="openai", base_url=None, key_env="LLM_API_KEY", needs_key=False,
                              default_model=None, reasoning_style="effort_if_set"),
}

# When no provider is named, the first one whose API key is set wins.
AUTO_DETECT_ORDER = ["anthropic", "openai", "openrouter", "groq", "gemini"]

# Paid prices in USD per 1M tokens (input, output), checked 2026-10-02: Claude from Anthropic's price list,
# the rest from the OpenRouter catalogue. ":free" variants are projected at the paid price, because the free
# tier can't carry production volume. Override with LLM_PRICE_IN / LLM_PRICE_OUT for any other model.
MODEL_PRICES = {
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-fable-5-1": (10.00, 50.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6-sol": (2.00, 10.00),
    "gpt-6-astra": (10.00, 50.00),
    "gpt-5.6-luna": (0.20, 1.20),
    "qwen/qwen3.8-27b": (0.42, 3.00),
    "google/gemma-4-31b-it": (0.09, 0.34),
    "nvidia/nemotron-3.5-lightning": (0.06, 0.16),
    "nvidia/nemotron-3-super-120b-a12b": (0.08, 0.45),
}


def price_for(model: str | None):
    """(USD per input token, USD per output token) for a model id, or None if unknown."""
    env_in, env_out = os.environ.get("LLM_PRICE_IN"), os.environ.get("LLM_PRICE_OUT")
    if env_in and env_out:
        try:
            return float(env_in) / 1_000_000, float(env_out) / 1_000_000
        except ValueError:
            pass
    if not model:
        return None
    name = model[:-5] if model.endswith(":free") else model
    candidates = [name, name.split("/", 1)[-1]]          # "openai/gpt-6-luna" -> "gpt-6-luna"
    for c in candidates:
        if c in MODEL_PRICES:
            p_in, p_out = MODEL_PRICES[c]
            return p_in / 1_000_000, p_out / 1_000_000
    for c in candidates:                                  # dated snapshots, e.g. "gpt-6-luna-2026-05-01"
        best = max((k for k in MODEL_PRICES if c.startswith(k + "-")), key=len, default=None)
        if best:
            p_in, p_out = MODEL_PRICES[best]
            return p_in / 1_000_000, p_out / 1_000_000
    return None
