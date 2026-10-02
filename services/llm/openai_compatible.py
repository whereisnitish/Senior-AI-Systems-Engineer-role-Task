"""Client for any OpenAI-compatible /chat/completions endpoint: OpenAI, OpenRouter, Groq, Gemini, Ollama, vLLM, ...

Standard library only. Endpoints differ in which optional parameters they accept (JSON mode, temperature,
reasoning effort, max_tokens vs max_completion_tokens), so when a request is rejected for naming one of them
the client drops or renames it, remembers that, and retries.
"""
from __future__ import annotations

import json
import random
import re
import time
import urllib.error
import urllib.request

from services.llm.base import BaseLLMClient, LLMError

# Optional parameters the client is willing to drop when an endpoint rejects them, in the order they're checked.
DROPPABLE = ("response_format", "temperature", "reasoning_effort", "reasoning")
# Error text meaning retries can't help: the quota or the credit has run out.
QUOTA_MARKERS = ("per-day", "insufficient_quota", "quota exceeded", "exceeded your current quota", "credit")


class OpenAICompatibleClient(BaseLLMClient):
    def __init__(self, api_key: str | None, model: str, base_url: str, *, provider: str = "openai-compatible",
                 key_env: str | None = None, headers: dict | None = None, reasoning_style: str = "effort_if_set",
                 send_temperature: bool = True, token_param: str = "max_tokens", **kw):
        super().__init__(model, provider=provider, **kw)
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.api_key = api_key
        self.key_env = key_env
        self.headers = headers or {}
        self.reasoning_style = reasoning_style
        self.token_param = token_param
        if not send_temperature:
            self._dropped.add("temperature")

    def _payload(self, messages: list) -> dict:
        p = {"model": self.model, "messages": messages, self.token_param: self.max_tokens}
        if "temperature" not in self._dropped:
            p["temperature"] = 0
        style, level = self.reasoning_style, self.reasoning
        if style == "openrouter" and "reasoning" not in self._dropped:
            # a classification label does not need thousands of thinking tokens
            p["reasoning"] = {"enabled": False} if level == "off" else {"effort": level, "exclude": True}
        elif style == "openai" and "reasoning_effort" not in self._dropped:
            p["reasoning_effort"] = "low" if level == "off" else level
        elif style == "effort_if_set" and level != "off" and "reasoning_effort" not in self._dropped:
            p["reasoning_effort"] = level
        if "response_format" not in self._dropped:
            p["response_format"] = {"type": "json_object"}
        return p

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=json.dumps(payload).encode("utf-8"), method="POST", headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _adapt(self, payload: dict, low: str) -> bool:
        """The endpoint rejected the request: drop or rename the parameter it complained about. True = retry."""
        for name in DROPPABLE:
            if name in payload and re.search(rf"\b{name}\b", low):
                return self._drop_param(name)
        if "max_tokens" in payload and re.search(r"\bmax_tokens\b", low):
            with self._lock:
                self.token_param = "max_completion_tokens"
            self._log(f"  {self.provider}/{self.model}: using max_completion_tokens instead of max_tokens")
            return True
        # OpenRouter: "No endpoints found that support the requested parameters" (without naming one)
        if "response_format" in payload and "support" in low:
            return self._drop_param("response_format")
        return False

    def _chat(self, messages: list) -> tuple[str, dict, str]:
        model = self.model
        last_err = "no attempt made"
        attempt = adaptations = 0
        while attempt < self.max_retries:
            attempt += 1
            self._throttle()
            payload = self._payload(messages)
            try:
                data = self._post(payload)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:400]
                last_err = f"{self.provider}/{model}: HTTP {e.code} {body}"
                low = body.lower()
                if e.code in (400, 404, 422) and adaptations < 6 and self._adapt(payload, low):
                    adaptations += 1
                    attempt -= 1
                    continue
                if e.code in (401, 403):
                    hint = f"; check {self.key_env}" if self.key_env else ""
                    self._stop_for_run(f"auth failed ({e.code})")
                    raise LLMError(f"auth failed ({e.code}){hint}")
                if e.code == 402 or (e.code == 429 and any(m in low for m in QUOTA_MARKERS)):
                    # quota or credit exhausted: no amount of backoff helps; stop calling for the rest of the run
                    self._stop_for_run(f"{self.provider} quota exhausted")
                    raise LLMError(f"{self.provider} quota exhausted (HTTP {e.code}); free tiers reset daily")
                if e.code == 404 and "model" in low:
                    self._stop_for_run(f"model '{model}' not found")
                    raise LLMError(f"model '{model}' not found at {self.provider}")
                if e.code in (408, 409, 429, 500, 502, 503, 504):
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt * 2)
                    self._log(f"  {self.provider}/{model} HTTP {e.code}, retry in {delay:.0f}s")
                    time.sleep(delay + random.random())
                    continue
                raise LLMError(last_err)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = f"{self.provider}/{model}: network {e}"
                time.sleep(min(30, 2 ** attempt))
                continue
            except json.JSONDecodeError:
                last_err = f"{self.provider}/{model}: non-JSON HTTP body"
                continue
            if "error" in data:                         # OpenRouter sometimes returns 200 with an error body
                last_err = f"{self.provider}/{model}: {str(data['error'])[:300]}"
                code = data["error"].get("code") if isinstance(data["error"], dict) else None
                if code in (429, 502, 503):
                    time.sleep(min(60, 2 ** attempt * 2) + random.random())
                    continue
                raise LLMError(last_err)
            try:
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError):
                last_err = f"{self.provider}/{model}: malformed response"
                continue
            finish = choice.get("finish_reason") or ""
            if not content.strip():
                last_err = f"{self.provider}/{model}: empty content (finish_reason={finish})"
                if finish == "length":    # spent the whole budget reasoning; retrying won't help
                    break
                continue
            usage = dict(data.get("usage") or {})
            usage["finish_reason"] = finish
            return content, usage, data.get("model") or model
        raise LLMError(last_err)
