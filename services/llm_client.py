"""OpenRouter chat-completions client: retries, circuit breaker, quota and JSON-mode handling."""
from __future__ import annotations

import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request

from utils.config import OPENROUTER_URL


class LLMError(Exception):
    pass


class OpenRouterClient:
    """Single-model client. Retries transient failures; never switches to another model."""

    def __init__(self, api_key: str, model: str, timeout: int = 120, max_retries: int = 3,
                 min_interval: float = 0.0, verbose: bool = False, reasoning: str = "off",
                 max_tokens: int = 8000):
        self.api_key = api_key
        self.model = model
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.verbose = verbose
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._json_mode = True
        # circuit breaker: after N consecutive failed messages, stop calling for a cool-down period
        self._fail_streak = 0
        self._open_until = 0.0
        self._open_reason = ""
        self.breaker_threshold = 3
        self.breaker_cooldown = 120.0

    def _record(self, ok: bool):
        with self._lock:
            if ok:
                self._fail_streak = 0
            else:
                self._fail_streak += 1
                if self._fail_streak >= self.breaker_threshold and self._open_until < time.monotonic():
                    self._open(self.breaker_cooldown, f"{self._fail_streak} consecutive LLM failures")

    def _open(self, seconds: float, why: str):
        self._open_until = time.monotonic() + seconds
        self._open_reason = why
        span = "for the rest of this run" if seconds == float("inf") else f"for {seconds:.0f}s"
        self._log(f"  circuit open: {why}; rules-only {span}")

    def _throttle(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _post(self, payload: dict) -> dict:
        req = urllib.request.Request(
            OPENROUTER_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                     "X-Title": "message-triage"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat(self, messages: list) -> tuple[str, dict, str]:
        """Returns (content, usage, model_used). Raises LLMError when every retry fails."""
        if time.monotonic() < self._open_until:
            raise LLMError(f"circuit open ({self._open_reason}); skipped LLM call")
        try:
            result = self._chat(messages)
        except LLMError:
            self._record(False)
            raise
        self._record(True)
        return result

    def _payload(self, messages: list) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": 0, "max_tokens": self.max_tokens}
        if self.reasoning == "off":     # a classification label does not need thousands of thinking tokens
            payload["reasoning"] = {"enabled": False}
        else:
            payload["reasoning"] = {"effort": self.reasoning, "exclude": True}
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _chat(self, messages: list) -> tuple[str, dict, str]:
        model = self.model
        last_err = "no attempt made"
        attempt = 0
        while attempt < self.max_retries:
            attempt += 1
            self._throttle()
            payload = self._payload(messages)
            try:
                data = self._post(payload)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "replace")[:400]
                last_err = f"{model}: HTTP {e.code} {body}"
                low = body.lower()
                if e.code in (400, 404, 422) and "response_format" in payload and ("response_format" in low or "support" in low):
                    self._json_mode = False          # endpoint lacks JSON mode: rely on the prompt + parser
                    self._log(f"  {model}: JSON mode unsupported, continuing without it")
                    attempt -= 1
                    continue
                if e.code in (401, 403):
                    raise LLMError(f"auth failed ({e.code}); check OPENROUTER_API_KEY")
                if e.code == 429 and "per-day" in low:
                    # daily quota: no amount of backoff helps; stop calling for the rest of the run
                    self._open(float("inf"), "daily free-model quota exhausted")
                    raise LLMError("daily free-model quota exhausted (resets 00:00 UTC)")
                if e.code in (429, 500, 502, 503, 504):
                    retry_after = e.headers.get("Retry-After") if e.headers else None
                    delay = float(retry_after) if retry_after and retry_after.isdigit() else min(60, 2 ** attempt * 2)
                    self._log(f"  {model} HTTP {e.code}, retry in {delay:.0f}s")
                    time.sleep(delay + random.random())
                    continue
                raise LLMError(last_err)
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last_err = f"{model}: network {e}"
                time.sleep(min(30, 2 ** attempt))
                continue
            except json.JSONDecodeError:
                last_err = f"{model}: non-JSON HTTP body"
                continue
            if "error" in data:                         # OpenRouter sometimes returns 200 with an error body
                last_err = f"{model}: {str(data['error'])[:300]}"
                code = data["error"].get("code") if isinstance(data["error"], dict) else None
                if code in (429, 502, 503):
                    time.sleep(min(60, 2 ** attempt * 2) + random.random())
                    continue
                raise LLMError(last_err)
            try:
                choice = data["choices"][0]
                content = choice["message"].get("content") or ""
            except (KeyError, IndexError, TypeError, AttributeError):
                last_err = f"{model}: malformed response"
                continue
            finish = choice.get("finish_reason") or ""
            if not content.strip():
                last_err = f"{model}: empty content (finish_reason={finish})"
                if finish == "length":    # spent the whole budget reasoning; retrying won't help
                    break
                continue
            usage = dict(data.get("usage") or {})
            usage["finish_reason"] = finish
            return content, usage, data.get("model") or model
        raise LLMError(last_err)

    def _log(self, msg):
        if self.verbose:
            print(msg, file=sys.stderr)
