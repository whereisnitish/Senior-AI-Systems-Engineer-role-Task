#!/usr/bin/env python3
"""
Message triage CLI.

    python triage.py messages.json -o results.json

Pipeline per message:
    1. normalise   - coerce whatever is in the file into a safe record (never raises)
    2. pre-checks  - deterministic rules: empty text, injection patterns, regex entities
    3. LLM         - classify intents + extract entities (any provider, JSON output)
    4. validate    - schema-check the LLM output, drop hallucinated entities
    5. policy      - code (not the model) decides route, priority, confidence, needs_human

The LLM only ever *labels*. Every action, handler and the human/automation decision
comes from the policy table, so a message cannot talk its way into an action.

Layout:
    triage.py                    command-line entry point (this file)
    services/triage_service.py   per-message pipeline and the concurrent batch runner
    services/policy.py           business rules: route, priority, confidence, needs_human
    services/llm/factory.py      picks the provider (OpenAI, Claude, OpenRouter, Groq, Gemini, Ollama, any
                                 OpenAI-compatible URL) from --provider / env / whichever API key is set
    services/llm/base.py         shared client behaviour: pacing, circuit breaker, learning rejected params
    services/llm/openai_compatible.py   OpenAI-protocol client (standard library)
    services/llm/anthropic_client.py    Claude client (official anthropic SDK, optional dependency)
    utils/providers.py           provider presets, auto-detect order, per-model prices
    utils/config.py              thresholds and limits
    utils/taxonomy.py            brands, intents, risk flags, the policy table
    utils/prompts.py             system prompt and message wrapping
    utils/rules.py               regex pre-checks, entity extraction, keyword fallback
    utils/normalise.py           turns any input record into a safe one
    utils/llm_output.py          parses and validates model output
    utils/cache.py               on-disk response cache
    utils/files.py               .env, input and output files
    utils/reporting.py           progress lines and the summary table

Standard library only, Python 3.9+. The Claude provider additionally needs `pip install anthropic` (Python 3.10+).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from services.llm import BaseLLMClient, LLMConfigError, create_client, resolve
from services.triage_service import run_batch
from utils.cache import Cache
from utils.files import InputError, load_env, load_input, write_results
from utils.normalise import normalise
from utils.providers import AUTO_DETECT_ORDER, PROVIDERS
from utils.reporting import print_progress, print_summary

HERE = Path(__file__).resolve().parent


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Triage inbound customer messages.")
    ap.add_argument("input", nargs="?", default="messages.json", help="JSON array (or JSON Lines) of messages")
    ap.add_argument("-o", "--output", default="results.json", help="where to write results (default results.json)")
    ap.add_argument("--provider", choices=list(PROVIDERS),
                    help="LLM provider (default: LLM_PROVIDER, else whichever API key is set)")
    ap.add_argument("--model", help="model id (default: LLM_MODEL, else the provider's default)")
    ap.add_argument("--workers", type=int, default=2, help="concurrent LLM calls (keep low on free tier)")
    ap.add_argument("--reasoning", default="off", choices=["off", "low", "medium", "high"],
                    help="model thinking: off (default) or an effort level")
    ap.add_argument("--max-tokens", type=int, default=8000,
                    help="output budget incl. reasoning tokens (reasoning models need several thousand)")
    ap.add_argument("--min-interval", type=float, default=3.0,
                    help="min seconds between LLM requests (free tier ~20 req/min); 0 to disable")
    ap.add_argument("--rules-only", action="store_true", help="skip the LLM entirely (degraded mode)")
    ap.add_argument("--no-cache", action="store_true", help="don't read/write the .triage_cache directory")
    ap.add_argument("--limit", type=int, default=0, help="only process the first N messages")
    ap.add_argument("-q", "--quiet", action="store_true")
    return ap.parse_args(argv)


def build_client(args) -> BaseLLMClient | None:
    if args.rules_only:
        print("rules-only mode: LLM disabled", file=sys.stderr)
        return None
    try:
        cfg = resolve(args.provider, args.model)
        if cfg is None:
            keys = ", ".join(PROVIDERS[p]["key_env"] for p in AUTO_DETECT_ORDER)
            print(f"warning: no LLM configured (looked for {keys}, LLM_PROVIDER, LLM_BASE_URL); "
                  f"running rules-only (everything goes to a human)", file=sys.stderr)
            return None
        client = create_client(cfg, min_interval=args.min_interval, reasoning=args.reasoning,
                               max_tokens=args.max_tokens, verbose=not args.quiet)
    except LLMConfigError as e:
        sys.exit(f"error: {e}")
    print(f"LLM: {client.describe()}, chosen by {cfg['how']}", file=sys.stderr)
    if cfg["note"]:
        print(f"note: {cfg['note']}", file=sys.stderr)
    return client


def main(argv=None):
    args = parse_args(argv)
    load_env(HERE / ".env")
    load_env(Path.cwd() / ".env")

    try:
        raw = load_input(Path(args.input))
    except InputError as e:
        sys.exit(f"error: {e}")
    if args.limit:
        raw = raw[:args.limit]
    seen = set()
    records = [normalise(r, i, seen) for i, r in enumerate(raw)]

    client = build_client(args)
    cache = Cache(None if args.no_cache else HERE / ".triage_cache")

    t0 = time.time()
    results = run_batch(records, client, cache, workers=args.workers,
                        on_result=None if args.quiet else print_progress)

    out = Path(args.output)
    write_results(out, results)
    if not args.quiet:
        print_summary(results, time.time() - t0)
    print(f"wrote {len(results)} results to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
