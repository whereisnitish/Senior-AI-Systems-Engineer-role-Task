# Message triage

A command-line tool that reads inbound customer messages (WhatsApp, email, Instagram) for three brands and decides, for each one:

- **what the customer wants**: one or more typed intents, because a single message often holds more than one request
- **what can be pulled out of it**: order IDs, booking refs, phones, amounts, dates, products, people, language
- **what happens next and who handles it**: a next action and handler queue for each intent, a message-level priority, a confidence score, and a `needs_human` decision with reasons

The decision log is in [DECISION_LOG.md](DECISION_LOG.md).

## Run it

Requires Python 3.9+. It uses **only the standard library**, so there is nothing to `pip install`.
The model is `qwen/qwen3.8-27b:free` on OpenRouter. The tool only ever calls this one model and never falls back to another.

```bash
# 1. API key: put this in .env next to triage.py (or export it)
OPENROUTER_API_KEY=sk-or-...

# 2. Run
python triage.py messages.json -o results.json

# No key / offline: deterministic rules only (every message goes to a human, pre-labelled)
python triage.py messages.json --rules-only
```

A per-message table, the token usage and a projected cost are printed to stderr. Full results go to `results.json`.

**Free-tier note:** an OpenRouter account without credits gets **50 free-model requests a day**. One full run of this file uses about 23 requests, plus any retries. When the quota runs out, the tool stops calling the API straight away (no retry storm), and the remaining messages go to rules-only mode, flagged `llm_unavailable`. Results are cached in `.triage_cache/`, so the next run only calls the model for messages not yet classified.

| Option | Default | Purpose |
|---|---|---|
| `--model` | `qwen/qwen3.8-27b:free` | OpenRouter model id (or set `TRIAGE_MODEL`) |
| `--reasoning` | `off` | Model thinking. `off` sends `reasoning.enabled=false`; `low`/`medium`/`high` set an effort level |
| `--max-tokens` | 8000 | Output budget, including any reasoning tokens |
| `--workers` | 2 | Concurrent LLM calls |
| `--min-interval` | 3.0 | Minimum seconds between requests (free tier allows about 20 per minute) |
| `--rules-only` | off | Skip the LLM |
| `--no-cache` | off | Ignore `.triage_cache/`. Normally a rerun doesn't spend quota on messages already classified |
| `--limit N` | 0 (all) | Only process the first N messages |

Input can be a JSON array, a `{"messages": [...]}` object, or JSON Lines. `samples/edge_cases.json` and `samples/broken.jsonl` are deliberately broken inputs that you can run to see how the tool degrades.

## Tests (no API key needed)

```bash
python tests/test_llm_mock.py                          # LLM path against a local mock OpenRouter
python triage.py samples/edge_cases.json --rules-only  # malformed records
python triage.py samples/broken.jsonl --rules-only     # broken JSON Lines
```

The mock returns a deliberately **wrong** label (`order_status`) and a made-up booking ref for every message. The test checks that the code still sends the injection (005), the health question (019) and the stranded traveller (016) to the right humans, and drops the made-up entity. It also covers the free endpoint rejecting `response_format`, the daily quota running out (stops after 1 request instead of backing off), and output cut off at `max_tokens`. The API URL can be overridden with `OPENROUTER_URL`.

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
3. **LLM.** It classifies into a fixed set of intent labels and extracts entities as JSON. The message is passed inside `<customer_message>` tags as untrusted data. The model **only labels**: it has no tools, and nothing it writes is executed.
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
  llm_client.py              OpenRouter client: retries, circuit breaker, quota and JSON-mode handling
utils/
  config.py                  model, endpoint, pricing, thresholds
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
