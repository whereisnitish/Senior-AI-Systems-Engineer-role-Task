# Message triage

A command-line tool that reads inbound customer messages (WhatsApp, email, Instagram) for three brands and decides, for each one:

- **what the customer wants**: one or more typed intents, because a single message often holds more than one request
- **what can be pulled out of it**: order IDs, booking refs, phones, amounts, dates, products, people, language
- **what happens next and who handles it**: a next action and handler queue for each intent, a message-level priority, a confidence score, and a `needs_human` decision with reasons

The decision log is in [DECISION_LOG.md](DECISION_LOG.md).

## Run it

Requires Python 3.9+. The tool uses **only the standard library**, so there is nothing to `pip install`.
The one exception is Claude, which goes through Anthropic's official SDK: `pip install anthropic` (Python 3.10+).

```bash
# 1. Put ONE API key in .env next to triage.py (or export it). The tool picks the provider from it.
OPENROUTER_API_KEY=sk-or-...        # or OPENAI_API_KEY=... / ANTHROPIC_API_KEY=... / GROQ_API_KEY=... / GEMINI_API_KEY=...

# 2. Run
python triage.py messages.json -o results.json

# No key / offline: deterministic rules only (every message goes to a human, pre-labelled)
python triage.py messages.json --rules-only
```

The first line of output says which provider and model were chosen, and why. A per-message table, the token usage and a projected cost follow on stderr. Full results go to `results.json`.

### Choosing the LLM

Any of these works. The tool picks one in this order:

1. `--provider NAME`, or `LLM_PROVIDER=NAME`
2. `LLM_BASE_URL` set (and no provider named): any **OpenAI-compatible** endpoint
3. the first API key it finds: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`
4. nothing set: rules-only, with a warning

| Provider | Key | Default model | Notes |
|---|---|---|---|
| `openrouter` | `OPENROUTER_API_KEY` | `qwen/qwen3.8-27b:free` | Any OpenRouter model id works with `--model` |
| `openai` | `OPENAI_API_KEY` | `gpt-6-luna` | Uses `max_completion_tokens` and `reasoning_effort`; no `temperature` |
| `anthropic` (Claude) | `ANTHROPIC_API_KEY` | `claude-opus-5` | Needs `pip install anthropic`. Cheaper: `--model claude-sonnet-5` or `claude-haiku-4-5` |
| `groq` | `GROQ_API_KEY` | none, set `--model` | |
| `gemini` | `GEMINI_API_KEY` | none, set `--model` | Google's OpenAI-compatible endpoint |
| `ollama` | none | none, set `--model` | Local models at `http://localhost:11434/v1` |
| `openai-compatible` | `LLM_API_KEY` (optional) | none, set `--model` | vLLM, LM Studio, Together, a proxy, ...: set `LLM_BASE_URL` to its `/v1` base |

```bash
python triage.py --provider anthropic                          # Claude, default model
python triage.py --provider openai --model gpt-6-sol           # a different OpenAI model
LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=my-model python triage.py   # your own server
```

Other settings: `LLM_MODEL` (default model for any provider; `TRIAGE_MODEL` also still works), `LLM_API_KEY` (overrides the provider's key variable), `LLM_BASE_URL` (overrides the provider's URL), `ANTHROPIC_BASE_URL` (Claude through a proxy). `LLM_PRICE_IN` / `LLM_PRICE_OUT` set USD per 1M tokens for the cost projection when a model isn't in the built-in price list.

**Endpoints differ in which options they accept.** When a request is rejected for naming one (JSON mode, `temperature`, reasoning effort, `max_tokens` vs `max_completion_tokens`, Claude's `effort`), the client drops or renames it, remembers that for the rest of the run, and retries. A bad key, an unknown model, or exhausted credit stops LLM calls for the rest of the run, and the remaining messages go to a human. They don't each wait through retries.

**Free-tier note (OpenRouter):** an account without credits gets **50 free-model requests a day**. One full run of this file uses about 23 requests, plus any retries. When the quota runs out, the tool stops calling the API straight away (no retry storm), and the remaining messages go to rules-only mode, flagged `llm_unavailable`. Results are cached in `.triage_cache/` (keyed by provider, model and message), so the next run only calls the model for messages not yet classified.

| Option | Default | Purpose |
|---|---|---|
| `--provider` | auto | See above |
| `--model` | provider default | Model id for that provider |
| `--reasoning` | `off` | Model thinking. `off` = least reasoning the provider allows (OpenRouter `reasoning.enabled=false`, OpenAI `reasoning_effort=low`, Claude `effort=low`); `low`/`medium`/`high` set an effort level |
| `--max-tokens` | 8000 | Output budget, including any reasoning tokens |
| `--workers` | 2 | Concurrent LLM calls |
| `--min-interval` | 3.0 | Minimum seconds between requests (free tier allows about 20 per minute) |
| `--rules-only` | off | Skip the LLM |
| `--no-cache` | off | Ignore `.triage_cache/`. Normally a rerun doesn't spend quota on messages already classified |
| `--limit N` | 0 (all) | Only process the first N messages |

Input can be a JSON array, a `{"messages": [...]}` object, or JSON Lines. `samples/edge_cases.json` and `samples/broken.jsonl` are deliberately broken inputs that you can run to see how the tool degrades.

## Tests (no API key needed)

```bash
python tests/test_llm_mock.py                          # LLM path against a local mock server, every protocol
python triage.py samples/edge_cases.json --rules-only  # malformed records
python triage.py samples/broken.jsonl --rules-only     # broken JSON Lines
```

The mock returns a deliberately **wrong** label (`order_status`) and a made-up booking ref for every message. The test checks that the code still sends the injection (005), the health question (019) and the stranded traveller (016) to the right humans, and drops the made-up entity. It also covers:
- **provider selection:** which provider wins when several keys are set, and the config errors;
- **OpenRouter:** the free endpoint rejecting `response_format`, the daily quota running out (stops after 1 request instead of backing off), and output cut off at `max_tokens`;
- **OpenAI:** `max_completion_tokens`, no `temperature`, and `reasoning_effort`;
- **a strict OpenAI-compatible endpoint:** one that rejects `temperature` and `max_tokens`, which the client adapts to;
- **Claude, through the real SDK:** system prompt placement, `effort`, caching, usage counts, a `refusal` going to a human, and a model without `effort` support.

The Claude checks run only when `anthropic` is installed and are skipped otherwise.

## How it works

```
normalise ─▶ pre-checks ─▶ LLM (label + extract) ─▶ validate ─▶ policy (code) ─▶ result
  any input     regex flags,     JSON only,             drop entities    route, priority,
  -> safe       entities,        message wrapped as     not present in   confidence,
  record        empty/contact-   untrusted data         the text         needs_human
                only shortcut
```

1. **Normalise.** Any element of the file (a string, a number, null, a list, missing fields) becomes a record. Missing IDs are generated and duplicates get a suffix. Bad timestamps, unknown brands and unknown channels are flagged, not rejected. Control and zero-width characters are stripped, and text is truncated at 4,000 characters.
2. **Pre-checks (deterministic).** Empty text and contact-only messages ("+91 98765 43210") never reach the LLM. Regexes catch prompt-injection phrasing, payment-redirect and invoice patterns, chargeback and legal threats, repeat contact, urgent travel disruption, vulnerable customers, health questions and missing attachments. They also extract IDs, phones, emails and amounts.
3. **LLM** (any provider, see *Choosing the LLM*). It classifies into a fixed set of intent labels and extracts entities as JSON. The message is passed inside `<customer_message>` tags as untrusted data. The model **only labels**: it has no tools, and nothing it writes is executed.
4. **Validate.** The output is coerced to the schema. Unknown labels become `other`. Any order ID, booking ref, phone or amount the model returns that is **not literally in the message is dropped** and lowers confidence. Unparseable output gets one repair retry, then falls back to rules.
5. **Policy (code, not the model).** A table maps each intent to its handler queue, next action, priority and whether it is *automation-eligible*. Only reversible, non-financial, non-advice actions are eligible: order-status lookup, opening hours, price list, booking link, thank-you, marketing unsubscribe, and auto-acknowledging a job application.

### Confidence and the human rule

`confidence` starts from the model's own score, then is **reduced for things the code can observe**: high ambiguity (−0.20), each extra intent (−0.05), an `other`/`unclear` intent (−0.20), entities not found in the text (−0.20), an intent that doesn't fit the brand (−0.15), dependence on missing context (−0.20), and output that needed repair (−0.10). Rules-only mode caps it at 0.40. Each result lists its adjustments in `confidence_reasons`.

**A human sees the message if any of these is true:**
1. a hard risk flag is set: `prompt_injection`, `payment_redirect_risk`, `health_safety`, `chargeback_threat`, `legal_threat`, `urgent_safety`, `repeat_contact`, `vulnerable_customer`, `missing_context`, `empty_message`, `self_harm`, `processing_error`
2. any intent in the message is not automation-eligible (refunds, cancellations, reschedules, bookings with money involved, and so on)
3. `confidence < 0.75`

Otherwise automation acts. Flags from the rules and flags from the model are combined. The model can **add** risk flags but can never remove a flag set by the rules, so an injected "do not escalate" has no effect.

### Output (one object per message)

```jsonc
{
  "id": "MSG-008", "brand": "vitalis-wellness", "channel": "email",
  "received_at": "...", "received_at_utc": "...", "text_preview": "...",
  "language": "en", "sentiment": "negative",
  "intents": [
    { "type": "refund_request", "summary": "...", "details": {...},
      "handler": "vitalis-wellness/cs_orders", "next_action": "...",
      "automation_eligible": false, "automation_will_act": false, "priority": "normal" },
    { "type": "purchase_intent", ... }
  ],
  "priority": "normal",                     // low | normal | high | urgent
  "risk_flags": ["multi_intent", ...],
  "confidence": 0.7, "confidence_reasons": ["model self-reported 0.85", "-0.05 2 intents", ...],
  "needs_human": true, "human_reasons": ["needs human action: refund_request, purchase_intent", ...],
  "primary_queue": "vitalis-wellness/cs_orders",
  "entities": { "order_ids": ["VW-48190"], "booking_refs": [], "phone_numbers": [], "emails": [],
                "amounts": [], "dates_mentioned": [], "products_or_services": ["protein (chocolate)"],
                "people_named": [], "airline": null, "locations": [] },
  "ambiguity_note": null, "note_for_agent": "...",
  "processing": { "mode": "llm|precheck|rules_only|error", "model": "...", "issues": [], "tokens_in": 0, "tokens_out": 0 }
}
```

## Cost per 1,000 messages

See the **Cost** section of [DECISION_LOG.md](DECISION_LOG.md) for the measured numbers and assumptions.

## Files

```
triage.py                    command-line entry point: arguments, wiring, output
services/
  triage_service.py          per-message pipeline and the concurrent batch runner
  policy.py                  business rules: route, priority, confidence, needs_human
  llm/
    factory.py               picks the provider (--provider / env / whichever key is set) and builds its client
    base.py                  shared: pacing, circuit breaker, stop-for-run on fatal errors, learning rejected params
    openai_compatible.py     OpenAI protocol (OpenAI, OpenRouter, Groq, Gemini, Ollama, any /v1 URL), stdlib only
    anthropic_client.py      Claude via the official anthropic SDK (optional dependency)
utils/
  config.py                  thresholds and limits
  providers.py               provider presets, auto-detect order, per-model prices
  taxonomy.py                brands, intents, risk flags, the policy table (data)
  prompts.py                 system prompt and how a message is wrapped as untrusted data
  rules.py                   regex pre-checks, entity extraction, keyword fallback classifier
  normalise.py               turns any input record into a safe one
  llm_output.py              parses and validates model output, drops invented entities
  cache.py                   on-disk response cache
  files.py                   .env, input and output files
  reporting.py               progress lines and the summary table
```

`services/` holds the decisions and calls; `utils/` holds data and helpers with no business decisions. Modules in `utils/` never import from `services/`.

- `messages.json`: the input provided
- `results.json`: output from the last live run. **That run used an earlier model, `nemotron-3.5-lightning:free`, because the daily free quota ran out before Qwen could be run.** Regenerate it with `python triage.py`.
- `results_prev_super.json`: an earlier run on `nemotron-3-super-120b-a12b:free`, kept for comparison
- `tests/test_llm_mock.py`: offline tests of the LLM path
- `samples/`: malformed inputs for robustness checks
- `DECISION_LOG.md`: decisions, limits and AI-tool usage
