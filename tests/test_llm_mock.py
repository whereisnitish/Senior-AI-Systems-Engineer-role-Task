"""Exercise triage.py's LLM path against a local mock of OpenRouter (no API key or quota needed).

    python tests/test_llm_mock.py

The mock deliberately returns a WRONG label (order_status) plus a made-up booking ref for
every message, to show that the deterministic layers still catch injection, health questions
and hallucinated entities regardless of what the model says.
"""
import json, os, subprocess, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = tempfile.mkdtemp(prefix="triage_mock_")
seen = []
SCENARIO = {"name": "ok"}

GOOD = {"language": "en", "intents": [{"type": "order_status", "summary": "tracking", "details": {}}],
        "entities": {"order_ids": ["VW-48812"], "booking_refs": ["FAKE123"]}, "sentiment": "neutral",
        "risk_flags": [], "ambiguity": "none", "needs_prior_context": False, "confidence": 0.95, "note_for_agent": "x"}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen.append(body)
        sc = SCENARIO["name"]
        if sc == "quota":
            return self.send(429, {"error": {"message": "Rate limit exceeded: free-models-per-day.", "code": 429}})
        if sc == "no_json_mode" and "response_format" in body:
            return self.send(404, {"error": {"message": "No endpoints found that support the requested parameters: response_format", "code": 404}})
        if sc == "truncated":
            return self.send(200, {"choices": [{"message": {"content": '{"language": "en", "intents": [{"ty'}, "finish_reason": "length"}],
                                   "usage": {"prompt_tokens": 900, "completion_tokens": 8000}})
        content = "```json\n" + json.dumps(GOOD) + "\n```" if sc == "no_json_mode" else json.dumps(GOOD)
        self.send(200, {"model": body["model"], "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                        "usage": {"prompt_tokens": 950, "completion_tokens": 300}})


srv = HTTPServer(("127.0.0.1", 0), H)
threading.Thread(target=srv.serve_forever, daemon=True).start()
env = dict(os.environ, OPENROUTER_URL=f"http://127.0.0.1:{srv.server_port}/v1/chat/completions",
           OPENROUTER_API_KEY="sk-test")


def run(sc, limit=4):
    SCENARIO["name"] = sc; seen.clear()
    out = os.path.join(OUT, f"{sc}.json")
    p = subprocess.run([sys.executable, "triage.py", "messages.json", "-o", out, "--no-cache", "--limit", str(limit),
                        "--min-interval", "0", "-q"], cwd=PACK, env=env, capture_output=True, text=True, timeout=300)
    return p, json.load(open(out, encoding="utf-8"))


fails = 0
def check(name, cond, info=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  ({info})" if info and not cond else ""))
    fails += 0 if cond else 1


p, r = run("ok", limit=25)
check("ok: exit 0", p.returncode == 0, p.stderr[-300:])
check("ok: only the configured model is called", {b["model"] for b in seen} == {"qwen/qwen3.8-27b:free"}, {b["model"] for b in seen})
check("ok: reasoning disabled in payload", all(b.get("reasoning") == {"enabled": False} for b in seen), seen[0].get("reasoning"))
check("ok: 23 LLM + 2 precheck", sum(x["processing"]["mode"] == "llm" for x in r) == 23, [x["processing"]["mode"] for x in r])
check("ok: hallucinated booking ref dropped, real order id kept",
      "FAKE123" not in json.dumps(r[0]["entities"]) and "VW-48812" in r[0]["entities"]["order_ids"]
      and "unverified_entity" in r[0]["risk_flags"], r[0]["entities"])
m5 = next(x for x in r if x["id"] == "MSG-005")
check("ok: MSG-005 injection -> human/trust_safety despite model label",
      m5["needs_human"] and m5["primary_queue"].endswith("trust_safety") and "prompt_injection" in m5["risk_flags"])
m19 = next(x for x in r if x["id"] == "MSG-019")
check("ok: MSG-019 health question -> human despite model label", m19["needs_human"] and "health_safety" in m19["risk_flags"])
m16 = next(x for x in r if x["id"] == "MSG-016")
check("ok: MSG-016 stays urgent despite model label", m16["priority"] == "urgent")

p, r = run("no_json_mode")
check("no_json_mode: drops response_format, parses fenced JSON, no message lost",
      all(x["processing"]["mode"] == "llm" for x in r), [x["processing"]["issues"] for x in r])
check("no_json_mode: later calls omit response_format", "response_format" not in seen[-1])

p, r = run("quota", limit=25)
check("quota: exit 0, all 25 results present", p.returncode == 0 and len(r) == 25, p.stderr[-300:])
check("quota: fails fast (<=2 requests, no backoff storm)", len(seen) <= 2, len(seen))
check("quota: everything to a human", all(x["needs_human"] for x in r))
check("quota: reason recorded", "quota exhausted" in json.dumps(r[0]["processing"]["issues"]), r[0]["processing"]["issues"])

p, r = run("truncated", limit=2)
check("truncated: one request per message (no wasted repair retry)", len(seen) == 2, len(seen))
check("truncated: rules fallback with issue noted", all(x["processing"]["mode"] == "rules_only" and
      any("cut off" in i for i in x["processing"]["issues"]) for x in r), [x["processing"]["issues"] for x in r])

srv.shutdown()
print("\nALL PASS" if not fails else f"\n{fails} FAILED")
sys.exit(1 if fails else 0)
