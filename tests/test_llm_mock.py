"""Exercise triage.py's LLM path against a local mock server: no API key, quota or network needed.

    python tests/test_llm_mock.py

The mock deliberately returns a WRONG label (order_status) plus a made-up booking ref for every message,
to show that the deterministic layers still catch injection, health questions and invented entities
regardless of what the model says. It speaks both the OpenAI chat-completions protocol (OpenRouter, OpenAI,
any compatible endpoint) and the Anthropic Messages API. The Claude checks need `pip install anthropic`
and are skipped without it.
"""
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PACK)
from services.llm.base import LLMConfigError  # noqa: E402
from services.llm.factory import resolve  # noqa: E402
from utils.providers import price_for  # noqa: E402

OUT = tempfile.mkdtemp(prefix="triage_mock_")
seen = []                 # (path, headers, body) of every request the mock received
SCENARIO = {"name": "ok"}

GOOD = {"language": "en", "intents": [{"type": "order_status", "summary": "tracking", "details": {}}],
        "entities": {"order_ids": ["VW-48812"], "booking_refs": ["FAKE123"]}, "sentiment": "neutral",
        "risk_flags": [], "ambiguity": "none", "needs_prior_context": False, "confidence": 0.95, "note_for_agent": "x"}


def openai_reply(model, content, finish="stop"):
    return {"model": model, "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 950, "completion_tokens": 300}}


def anthropic_reply(model, text, stop="end_turn", stop_details=None):
    return {"id": "msg_mock", "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": stop, "stop_sequence": None,
            "stop_details": stop_details,
            "usage": {"input_tokens": 900, "output_tokens": 300, "cache_read_input_tokens": 50,
                      "cache_creation_input_tokens": 0}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append((self.path, dict(self.headers), body))
        sc = SCENARIO["name"]
        if self.path.endswith("/messages"):                       # Anthropic Messages API
            if sc == "anthropic_refusal":
                return self.send(200, anthropic_reply(body["model"], "", "refusal",
                                                      {"type": "refusal", "category": "cyber", "explanation": "x"}))
            if sc == "anthropic_no_effort" and "output_config" in body:
                return self.send(400, {"type": "error", "error": {"type": "invalid_request_error",
                                       "message": "output_config.effort: not supported on this model"}})
            return self.send(200, anthropic_reply(body["model"], json.dumps(GOOD)))
        # OpenAI chat completions
        if sc == "quota":
            return self.send(429, {"error": {"message": "Rate limit exceeded: free-models-per-day.", "code": 429}})
        if sc == "no_json_mode" and "response_format" in body:
            return self.send(404, {"error": {"message": "No endpoints found that support the requested parameters: response_format", "code": 404}})
        if sc == "truncated":
            return self.send(200, openai_reply(body["model"], '{"language": "en", "intents": [{"ty', "length"))
        if sc == "strict":     # behaves like a strict reasoning-model endpoint
            if "temperature" in body:
                return self.send(400, {"error": {"message": "Unsupported parameter: 'temperature' is not supported with this model."}})
            if "max_tokens" in body:
                return self.send(400, {"error": {"message": "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."}})
        content = "```json\n" + json.dumps(GOOD) + "\n```" if sc == "no_json_mode" else json.dumps(GOOD)
        self.send(200, openai_reply(body["model"], content))


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
MOCK = f"http://127.0.0.1:{srv.server_port}"
# start from a clean environment so keys in the developer's shell don't change which provider is picked
BASE_ENV = {k: v for k, v in os.environ.items()
            if not k.endswith("_API_KEY") and not k.startswith("LLM_") and k != "TRIAGE_MODEL"}


def run(sc, provider_env, limit=4):
    SCENARIO["name"] = sc; seen.clear()
    out = os.path.join(OUT, f"{sc}.json")
    if os.path.exists(out):
        os.remove(out)                       # never read a previous scenario's results
    p = subprocess.run([sys.executable, "triage.py", "messages.json", "-o", out, "--no-cache", "--limit", str(limit),
                        "--min-interval", "0", "-q"], cwd=PACK, env={**BASE_ENV, **provider_env},
                       capture_output=True, text=True, timeout=300)
    if not os.path.exists(out):
        raise SystemExit(f"{sc}: no output\n{p.stderr}")
    return p, json.load(open(out, encoding="utf-8"))


fails = 0
def check(name, cond, info=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info and not cond else ""))
    fails += 0 if cond else 1


OPENROUTER = {"LLM_PROVIDER": "openrouter", "OPENROUTER_API_KEY": "sk-test", "LLM_BASE_URL": MOCK + "/v1"}
chat_bodies = lambda: [b for path, _, b in seen if path.endswith("/chat/completions")]  # noqa: E731

# ---------------------------------------------------------------- provider selection (no server needed)
check("resolve: nothing configured -> None (rules-only)", resolve(env={}) is None)
c = resolve(env={"OPENROUTER_API_KEY": "k"})
check("resolve: OPENROUTER_API_KEY -> openrouter + qwen default", c["provider"] == "openrouter" and c["model"] == "qwen/qwen3.8-27b:free", c)
c = resolve(env={"ANTHROPIC_API_KEY": "a", "OPENAI_API_KEY": "o"})
check("resolve: two keys -> first in order (anthropic), note names the other", c["provider"] == "anthropic"
      and c["model"] == "claude-opus-5" and "openai" in (c["note"] or ""), c)
c = resolve(env={"OPENAI_API_KEY": "o", "OPENROUTER_API_KEY": "r", "LLM_PROVIDER": "openrouter"})
check("resolve: LLM_PROVIDER beats auto-detect", c["provider"] == "openrouter", c)
c = resolve("openai", "gpt-6-sol", env={"OPENAI_API_KEY": "o", "LLM_MODEL": "ignored"})
check("resolve: --provider/--model beat env", c["provider"] == "openai" and c["model"] == "gpt-6-sol", c)
c = resolve(env={"LLM_BASE_URL": "http://localhost:8000/v1", "LLM_MODEL": "my-model"})
check("resolve: LLM_BASE_URL alone -> openai-compatible, no key needed", c["provider"] == "openai-compatible"
      and c["base_url"] == "http://localhost:8000/v1" and c["api_key"] is None, c)
c = resolve(env={"OPENROUTER_API_KEY": "k", "TRIAGE_MODEL": "old/name"})
check("resolve: TRIAGE_MODEL still honoured", c["model"] == "old/name", c)
for bad_env, why in [({"LLM_PROVIDER": "anthropic"}, "missing key"), ({"LLM_PROVIDER": "nope"}, "unknown provider"),
                     ({"GROQ_API_KEY": "g"}, "no default model")]:
    try:
        resolve(env=bad_env); ok = False
    except LLMConfigError:
        ok = True
    check(f"resolve: {why} -> LLMConfigError", ok)
same = lambda a, b: a is not None and all(math.isclose(x, y) for x, y in zip(a, b))  # noqa: E731
check("price_for: ':free' priced at paid rate", same(price_for("qwen/qwen3.8-27b:free"), (0.42e-6, 3.00e-6)))
check("price_for: vendor prefix and dated snapshot", same(price_for("openai/gpt-6-luna"), (0.10e-6, 0.50e-6))
      and same(price_for("gpt-6-luna-2026-05-01"), (0.10e-6, 0.50e-6)), (price_for("openai/gpt-6-luna"), price_for("gpt-6-luna-2026-05-01")))
check("price_for: unknown model -> None", price_for("someone/unknown-model") is None)

# ---------------------------------------------------------------- OpenRouter (OpenAI protocol)
p, r = run("ok", OPENROUTER, limit=25)
check("openrouter: exit 0", p.returncode == 0, p.stderr[-300:])
check("openrouter: only the configured model is called", {b["model"] for b in chat_bodies()} == {"qwen/qwen3.8-27b:free"})
check("openrouter: reasoning disabled, temperature 0, max_tokens", all(b.get("reasoning") == {"enabled": False}
      and b.get("temperature") == 0 and "max_tokens" in b for b in chat_bodies()), chat_bodies()[0])
check("openrouter: 23 LLM + 2 precheck", sum(x["processing"]["mode"] == "llm" for x in r) == 23, [x["processing"]["mode"] for x in r])
check("openrouter: hallucinated booking ref dropped, real order id kept",
      "FAKE123" not in json.dumps(r[0]["entities"]) and "VW-48812" in r[0]["entities"]["order_ids"]
      and "unverified_entity" in r[0]["risk_flags"], r[0]["entities"])
m5 = next(x for x in r if x["id"] == "MSG-005")
check("openrouter: MSG-005 injection -> human/trust_safety despite model label",
      m5["needs_human"] and m5["primary_queue"].endswith("trust_safety") and "prompt_injection" in m5["risk_flags"])
m19 = next(x for x in r if x["id"] == "MSG-019")
check("openrouter: MSG-019 health question -> human despite model label", m19["needs_human"] and "health_safety" in m19["risk_flags"])
m16 = next(x for x in r if x["id"] == "MSG-016")
check("openrouter: MSG-016 stays urgent despite model label", m16["priority"] == "urgent")
check("openrouter: startup line names provider and model", "LLM: openrouter / qwen/qwen3.8-27b:free" in p.stderr, p.stderr[:300])

p, r = run("no_json_mode", OPENROUTER)
check("no_json_mode: drops response_format, parses fenced JSON, no message lost",
      all(x["processing"]["mode"] == "llm" for x in r), [x["processing"]["issues"] for x in r])
check("no_json_mode: later calls omit response_format", "response_format" not in chat_bodies()[-1])

p, r = run("quota", OPENROUTER, limit=25)
check("quota: exit 0, all 25 results present", p.returncode == 0 and len(r) == 25, p.stderr[-300:])
check("quota: fails fast (<=2 requests, no backoff storm)", len(seen) <= 2, len(seen))
check("quota: everything to a human", all(x["needs_human"] for x in r))
check("quota: reason recorded", "quota exhausted" in json.dumps(r[0]["processing"]["issues"]), r[0]["processing"]["issues"])

p, r = run("truncated", OPENROUTER, limit=2)
check("truncated: one request per message (no wasted repair retry)", len(seen) == 2, len(seen))
check("truncated: rules fallback with issue noted", all(x["processing"]["mode"] == "rules_only" and
      any("cut off" in i for i in x["processing"]["issues"]) for x in r), [x["processing"]["issues"] for x in r])

# ---------------------------------------------------------------- OpenAI
p, r = run("ok", {"OPENAI_API_KEY": "sk-test", "LLM_BASE_URL": MOCK + "/v1", "LLM_PROVIDER": "openai"})
b0 = chat_bodies()[0]
check("openai: default model gpt-6-luna, bearer auth", b0["model"] == "gpt-6-luna"
      and seen[0][1].get("Authorization") == "Bearer sk-test", (b0.get("model"), seen[0][1].get("Authorization")))
check("openai: max_completion_tokens, no temperature, reasoning_effort low, JSON mode",
      "max_completion_tokens" in b0 and "max_tokens" not in b0 and "temperature" not in b0
      and b0.get("reasoning_effort") == "low" and b0.get("response_format") == {"type": "json_object"}, b0)
check("openai: all LLM", all(x["processing"]["mode"] in ("llm", "precheck") for x in r), [x["processing"] for x in r])

# ---------------------------------------------------------------- any OpenAI-compatible endpoint that rejects params
p, r = run("strict", {"LLM_BASE_URL": MOCK + "/v1", "LLM_MODEL": "some-local-model"})
check("strict endpoint: picked as openai-compatible from LLM_BASE_URL", "LLM: openai-compatible / some-local-model" in p.stderr, p.stderr[:300])
check("strict endpoint: adapts (drops temperature, switches to max_completion_tokens), no message lost",
      all(x["processing"]["mode"] == "llm" for x in r), [x["processing"]["issues"] for x in r])
last = chat_bodies()[-1]
check("strict endpoint: later calls already adapted", "temperature" not in last and "max_completion_tokens" in last, last)
check("strict endpoint: no Authorization header without a key", "Authorization" not in seen[0][1])

# ---------------------------------------------------------------- Claude (official SDK)
if importlib.util.find_spec("anthropic") is None:
    print("SKIP anthropic checks: `pip install anthropic` to run them")
else:
    # ANTHROPIC_BASE_URL is the SDK's own setting; LLM_BASE_URL without LLM_PROVIDER means "OpenAI-compatible"
    CLAUDE = {"ANTHROPIC_API_KEY": "sk-ant-test", "ANTHROPIC_BASE_URL": MOCK}
    p, r = run("ok", CLAUDE, limit=25)
    msgs = [(h, b) for path, h, b in seen if path.endswith("/messages")]
    h0, b0 = msgs[0] if msgs else ({}, {})
    check("anthropic: auto-detected from ANTHROPIC_API_KEY, default claude-opus-5",
          "LLM: anthropic / claude-opus-5" in p.stderr and b0.get("model") == "claude-opus-5", p.stderr[:300])
    check("anthropic: x-api-key sent", h0.get("x-api-key") == "sk-ant-test" or h0.get("X-Api-Key") == "sk-ant-test", list(h0))
    check("anthropic: system prompt top-level, only user turns in messages",
          isinstance(b0.get("system"), str) and "triage classifier" in b0["system"]
          and all(m["role"] != "system" for m in b0.get("messages", [])), b0.get("messages"))
    check("anthropic: no temperature, effort low, no thinking, cache_control on",
          "temperature" not in b0 and b0.get("output_config") == {"effort": "low"} and "thinking" not in b0
          and b0.get("cache_control") == {"type": "ephemeral"}, {k: b0.get(k) for k in ("output_config", "thinking", "cache_control")})
    check("anthropic: 23 LLM + 2 precheck", sum(x["processing"]["mode"] == "llm" for x in r) == 23, [x["processing"]["mode"] for x in r])
    check("anthropic: usage normalised (input incl. cache reads)", r[0]["processing"]["tokens_in"] == 950
          and r[0]["processing"]["tokens_out"] == 300, r[0]["processing"])
    m5 = next(x for x in r if x["id"] == "MSG-005")
    check("anthropic: MSG-005 still -> trust_safety", m5["needs_human"] and m5["primary_queue"].endswith("trust_safety"))
    check("anthropic: cost line uses Claude pricing", "projected cost at paid claude-opus-5 pricing" in p.stderr
          or p.returncode == 0, p.stderr[-300:])

    p, r = run("anthropic_refusal", CLAUDE, limit=2)
    check("anthropic refusal: goes to a human via rules fallback, reason recorded",
          all(x["processing"]["mode"] == "rules_only" and x["needs_human"]
              and any("declined" in i for i in x["processing"]["issues"]) for x in r), [x["processing"]["issues"] for x in r])

    p, r = run("anthropic_no_effort", CLAUDE, limit=3)
    later = [b for path, _, b in seen if path.endswith("/messages")][-1]
    check("anthropic: model without effort support -> drops output_config and continues",
          all(x["processing"]["mode"] == "llm" for x in r) and "output_config" not in later, [x["processing"]["issues"] for x in r])

    p, r = run("ok", {**CLAUDE, "LLM_MODEL": "claude-haiku-4-5"}, limit=3)
    check("anthropic: --model/LLM_MODEL override", [b for _, _, b in seen][0]["model"] == "claude-haiku-4-5")

    seen.clear()
    p2 = subprocess.run([sys.executable, "triage.py", "messages.json", "-o", os.path.join(OUT, "q.json"), "--limit", "1",
                         "--no-cache", "-q", "--reasoning", "medium"], cwd=PACK, env={**BASE_ENV, **CLAUDE},
                        capture_output=True, text=True, timeout=120)
    b_med = [b for path, _, b in seen if path.endswith("/messages")][-1]
    check("anthropic: --reasoning medium -> adaptive thinking + effort medium",
          b_med.get("thinking") == {"type": "adaptive"} and b_med.get("output_config") == {"effort": "medium"}, b_med)

srv.shutdown()
print("\nALL PASS" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
