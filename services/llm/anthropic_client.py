"""Claude through the official Anthropic Python SDK (Messages API).

Optional dependency: `pip install anthropic` (needs Python 3.10+). It's imported only when this provider is used,
so the rest of the tool stays standard-library only. The SDK retries 408/409/429/5xx and connection errors itself.

Request choices for a classifier:
  - no temperature: current Claude models reject sampling parameters
  - "--reasoning off" is sent as output_config.effort = "low" rather than disabling thinking, which is what
    Anthropic recommends for classification; other levels also turn on adaptive thinking
  - top-level cache_control: the system prompt is identical on every call, so it's cached when long enough
  - a "refusal" stop is treated as a failed call, so the message goes to a human (no fallback to another model)
"""
from __future__ import annotations

from services.llm.base import BaseLLMClient, LLMConfigError, LLMError

# optional request parameters -> words in a 400 / TypeError message that mean "this one isn't supported here"
DROPPABLE = {"output_config": ("output_config", "effort"), "thinking": ("thinking",), "cache_control": ("cache_control",)}


class AnthropicClient(BaseLLMClient):
    def __init__(self, api_key: str, model: str, *, base_url: str | None = None, key_env: str = "ANTHROPIC_API_KEY", **kw):
        super().__init__(model, provider="anthropic", **kw)
        try:
            import anthropic
        except ImportError as e:
            raise LLMConfigError("provider 'anthropic' needs the official SDK: pip install anthropic") from e
        self._sdk = anthropic
        self.key_env = key_env
        opts = {"api_key": api_key, "timeout": float(self.timeout), "max_retries": self.max_retries}
        if base_url:
            opts["base_url"] = base_url
        self._client = anthropic.Anthropic(**opts)

    def _params(self, messages: list) -> dict:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        convo = [{"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"]
        p = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": convo}
        if "cache_control" not in self._dropped:
            p["cache_control"] = {"type": "ephemeral"}
        if "output_config" not in self._dropped:
            p["output_config"] = {"effort": "low" if self.reasoning == "off" else self.reasoning}
        if self.reasoning != "off" and "thinking" not in self._dropped:
            p["thinking"] = {"type": "adaptive"}
        return p

    def _adapt(self, params: dict, text: str) -> bool:
        low = text.lower()
        for name, words in DROPPABLE.items():
            if name in params and any(w in low for w in words):
                return self._drop_param(name)
        return False

    def _chat(self, messages: list) -> tuple[str, dict, str]:
        sdk = self._sdk
        for _ in range(len(DROPPABLE) + 1):
            params = self._params(messages)
            self._throttle()
            try:
                resp = self._client.messages.create(**params)
            except TypeError as e:                      # an older SDK that doesn't know an optional parameter
                if self._adapt(params, str(e)):
                    continue
                raise LLMError(f"anthropic SDK rejected the request: {e}") from e
            except (sdk.AuthenticationError, sdk.PermissionDeniedError) as e:
                self._stop_for_run(f"auth failed ({e.status_code})")
                raise LLMError(f"auth failed ({e.status_code}); check {self.key_env}") from e
            except sdk.NotFoundError as e:
                self._stop_for_run(f"model '{self.model}' not found")
                raise LLMError(f"model '{self.model}' not found at anthropic") from e
            except sdk.BadRequestError as e:
                msg = str(getattr(e, "message", "") or e)
                if "credit balance" in msg.lower():
                    self._stop_for_run("anthropic credit balance too low")
                    raise LLMError("anthropic credit balance too low") from e
                if self._adapt(params, msg):
                    continue
                raise LLMError(f"anthropic bad request: {msg[:300]}") from e
            except sdk.RateLimitError as e:
                raise LLMError("anthropic rate limit (after SDK retries)") from e
            except sdk.APIStatusError as e:             # 5xx / 529 overloaded, after SDK retries
                raise LLMError(f"anthropic HTTP {e.status_code}: {str(getattr(e, 'message', e))[:300]}") from e
            except (sdk.APITimeoutError, sdk.APIConnectionError) as e:
                raise LLMError(f"anthropic connection error: {e}") from e

            stop = resp.stop_reason
            if stop == "refusal":
                category = getattr(resp.stop_details, "category", None) if resp.stop_details else None
                raise LLMError(f"model declined the request (category={category}); routed to a human")
            text = "".join(b.text for b in resp.content if b.type == "text")
            finish = "length" if stop == "max_tokens" else (stop or "")
            if not text.strip():
                raise LLMError(f"anthropic/{self.model}: empty content (stop_reason={stop})")
            u = resp.usage
            prompt_tokens = (u.input_tokens or 0) + (u.cache_read_input_tokens or 0) + (u.cache_creation_input_tokens or 0)
            usage = {"prompt_tokens": prompt_tokens, "completion_tokens": u.output_tokens or 0, "finish_reason": finish}
            return text, usage, resp.model or self.model
        raise LLMError("anthropic: request kept being rejected after dropping optional parameters")
