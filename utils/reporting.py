"""Progress lines and the end-of-run summary table (to stderr)."""
from __future__ import annotations

import sys

from utils.config import PRICE_IN, PRICE_OUT, PRICED_MODEL
from utils.taxonomy import HARD_FLAGS


def print_progress(done: int, total: int, r: dict):
    print(f"[{done}/{total}] {r['id']}: {r.get('processing', {}).get('mode')} "
          f"-> {r.get('primary_queue')} (conf {r.get('confidence')})", file=sys.stderr)


def print_summary(results: list, elapsed: float):
    print(f"\n{'ID':<14}{'PRI':<8}{'CONF':<6}{'HUMAN':<7}{'QUEUE':<34}INTENTS / FLAGS", file=sys.stderr)
    for r in results:
        intents = ",".join(i["type"] for i in r.get("intents", []))
        flags = ",".join(f for f in r.get("risk_flags", []) if f in HARD_FLAGS)
        print(f"{str(r.get('id'))[:13]:<14}{r.get('priority',''):<8}{r.get('confidence',0):<6}"
              f"{'yes' if r.get('needs_human') else 'no':<7}{str(r.get('primary_queue'))[:33]:<34}"
              f"{intents}{'  [' + flags + ']' if flags else ''}", file=sys.stderr)

    n = len(results)
    tin = sum(r.get("processing", {}).get("tokens_in", 0) for r in results)
    tout = sum(r.get("processing", {}).get("tokens_out", 0) for r in results)
    llm_n = sum(1 for r in results if r.get("processing", {}).get("tokens_in"))
    modes = {}
    for r in results:
        m = r.get("processing", {}).get("mode")
        modes[m] = modes.get(m, 0) + 1
    print(f"\n{n} messages in {elapsed:.1f}s | modes: {modes} | needs_human: "
          f"{sum(1 for r in results if r.get('needs_human'))}/{n}", file=sys.stderr)
    if llm_n:
        per_msg = (tin * PRICE_IN + tout * PRICE_OUT) / llm_n
        print(f"tokens: {tin} in / {tout} out over {llm_n} LLM calls "
              f"(avg {tin / llm_n:.0f} in, {tout / llm_n:.0f} out)", file=sys.stderr)
        print(f"projected cost at paid {PRICED_MODEL} pricing: ${per_msg * 1000:.4f} per 1,000 messages, "
              f"${per_msg * 10000 * 30:.2f}/month at 10k/day", file=sys.stderr)
