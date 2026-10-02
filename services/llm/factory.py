"""Pick the LLM provider from the command line and environment, and build its client.

Order of precedence:
  1. --provider, or LLM_PROVIDER
  2. LLM_BASE_URL set               -> "openai-compatible" (any endpoint speaking that protocol)
  3. the first API key found, in AUTO_DETECT_ORDER (ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, ...)
  4. nothing configured            -> None (the caller runs rules-only)

Model: --model, then LLM_MODEL (or the older TRIAGE_MODEL), then the provider's default.
Key:   LLM_API_KEY overrides the provider's own key variable. Base URL: LLM_BASE_URL overrides the preset.
"""
from __future__ import annotations

import os

from services.llm.base import BaseLLMClient, LLMConfigError
from utils.providers import AUTO_DETECT_ORDER, PROVIDERS


def resolve(provider: str | None = None, model: str | None = None, env=None) -> dict | None:
    """Work out provider, model, key and base URL. Raises LLMConfigError on a mistake the operator must fix."""
    env = os.environ if env is None else env
    get = lambda k: (env.get(k) or "").strip() if k else ""   # noqa: E731
    name = (provider or get("LLM_PROVIDER")).lower() or None
    how, note = ("--provider" if provider else "LLM_PROVIDER"), None
    if not name and get("LLM_BASE_URL"):
        name, how = "openai-compatible", "LLM_BASE_URL"
    if not name:
        found = [p for p in AUTO_DETECT_ORDER if get(PROVIDERS[p]["key_env"])]
        if not found:
            return None
        name, how = found[0], f"found {PROVIDERS[found[0]]['key_env']}"
        if len(found) > 1:
            note = f"keys for {', '.join(found[1:])} are also set; set LLM_PROVIDER to choose another"
    if name not in PROVIDERS:
        raise LLMConfigError(f"unknown provider '{name}'; choose one of: {', '.join(PROVIDERS)}")

    preset = PROVIDERS[name]
    key = get("LLM_API_KEY") or get(preset["key_env"])
    if preset.get("needs_key", True) and not key:
        raise LLMConfigError(f"provider '{name}' needs {preset['key_env']} (or LLM_API_KEY)")
    model = model or get("LLM_MODEL") or get("TRIAGE_MODEL") or preset["default_model"]
    if not model:
        raise LLMConfigError(f"provider '{name}' has no default model; set LLM_MODEL or pass --model")
    base_url = get("LLM_BASE_URL") or preset["base_url"]
    if preset["protocol"] == "openai" and not base_url:
        raise LLMConfigError(f"provider '{name}' needs LLM_BASE_URL (the endpoint's /v1 base)")
    return {"provider": name, "model": model, "api_key": key or None, "base_url": base_url,
            "how": how, "note": note}


def create_client(cfg: dict, **kw) -> BaseLLMClient:
    """Build the client for a resolve() result. kw: reasoning, max_tokens, min_interval, verbose, ..."""
    preset = PROVIDERS[cfg["provider"]]
    if preset["protocol"] == "anthropic":
        from services.llm.anthropic_client import AnthropicClient
        return AnthropicClient(cfg["api_key"], cfg["model"], base_url=cfg["base_url"], key_env=preset["key_env"], **kw)
    from services.llm.openai_compatible import OpenAICompatibleClient
    return OpenAICompatibleClient(
        cfg["api_key"], cfg["model"], cfg["base_url"], provider=cfg["provider"], key_env=preset["key_env"],
        headers=preset.get("headers"), reasoning_style=preset.get("reasoning_style", "effort_if_set"),
        send_temperature=preset.get("send_temperature", True), token_param=preset.get("token_param", "max_tokens"),
        **kw)
