"""What every LLM client shares: request pacing, the circuit breaker, and learning which parameters an endpoint rejects."""
from __future__ import annotations

import sys
import threading
import time


class LLMError(Exception):
    """A call failed; the message goes to the rules-only fallback (and therefore to a human)."""


class LLMConfigError(Exception):
    """Misconfiguration the operator has to fix: unknown provider, missing key, model or SDK."""


class BaseLLMClient:
    """Subclasses implement _chat(messages) -> (content, usage, model_used).

    usage is normalised to {"prompt_tokens", "completion_tokens", "finish_reason"}, where finish_reason is
    "length" when the output hit max_tokens. messages use the OpenAI shape: [{"role": "system"|"user"|"assistant",
    "content": str}]; each client converts that to its own wire format.
    """

    def __init__(self, model: str, *, provider: str, reasoning: str = "off", max_tokens: int = 8000,
                 timeout: int = 120, max_retries: int = 3, min_interval: float = 0.0, verbose: bool = False):
        self.provider = provider
        self.model = model
        self.reasoning = reasoning
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self.verbose = verbose
        self._lock = threading.Lock()
        self._last_call = 0.0
        self._dropped = set()          # request parameters this endpoint has rejected; learned at runtime
        # circuit breaker: after N consecutive failed messages, stop calling for a cool-down period
        self._fail_streak = 0
        self._open_until = 0.0
        self._open_reason = ""
        self.breaker_threshold = 3
        self.breaker_cooldown = 120.0

    # ------------------------------------------------------------------ public
    def chat(self, messages: list) -> tuple[str, dict, str]:
        """Returns (content, usage, model_used). Raises LLMError when the call can't be completed."""
        if time.monotonic() < self._open_until:
            raise LLMError(f"circuit open ({self._open_reason}); skipped LLM call")
        try:
            result = self._chat(messages)
        except LLMError:
            self._record(False)
            raise
        self._record(True)
        return result

    def describe(self) -> str:
        return f"{self.provider} / {self.model} (reasoning {self.reasoning})"

    # ------------------------------------------------------------------ for subclasses
    def _chat(self, messages: list) -> tuple[str, dict, str]:
        raise NotImplementedError

    def _stop_for_run(self, why: str):
        """Errors no retry can fix (bad key, no credit, unknown model): skip the LLM for the rest of the run."""
        with self._lock:
            if self._open_until != float("inf"):
                self._open(float("inf"), why)

    def _drop_param(self, name: str) -> bool:
        """Remember that the endpoint rejected a parameter; the caller retries without it."""
        with self._lock:
            if name not in self._dropped:
                self._dropped.add(name)
                self._log(f"  {self.provider}/{self.model}: '{name}' not supported, continuing without it")
        return True

    def _throttle(self):
        if self.min_interval <= 0:
            return
        with self._lock:
            wait = self._last_call + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()

    def _log(self, msg):
        if self.verbose:
            print(msg, file=sys.stderr)

    # ------------------------------------------------------------------ internals
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
