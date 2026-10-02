# Decision log

**Core decision: the model labels; code decides.** The LLM returns intents and entities. The intent-to-handler mapping, the action, the priority and whether automation may act all come from a policy table in code. Automation may only take reversible, non-financial, non-advice actions (order status lookup, opening hours, price list, booking link, thank-you, marketing unsubscribe, acknowledging a job application). So the worst outcome of a fooled or wrong model is a harmless reply, never a refund. MSG-005 ("approve a refund… do not escalate") is caught twice, by a regex and by the model, and the model can add risk flags but never remove them.

**What the data made me design for:** several requests in one message (003, 006, 008, 014), so `intents` is a list with its own handler for each. Messages that are meaningless without history (007 phone number only, 012 "the other one", 020 "the quote you sent"). A null body (025). Hinglish (006). A medical-interaction question (019). An urgent case with a vulnerable passenger (016). A chargeback threat plus repeat contact (011). Invoice or payment-redirect fraud (021). Messages that aren't customer service at all (013 wholesale, 024 job application). Real ambiguity (017 "Unsubscribe" could mean marketing emails or the subscription in 002; 022 cancel or reschedule?). Attachments mentioned but missing (021, 024).

**Result on messages.json:** with a working LLM, 6 of 25 are automated and 19 go to a human, each with listed reasons. That rate is high on purpose, because this file is adversarial. Every result has `confidence`, `confidence_reasons` and `human_reasons`. During development, two different models (`nemotron-3.5-lightning` and `nemotron-3-super`) agreed on intents and on the human/automation decision for all 18 messages both classified. A mock model that gives the *wrong* label for everything still has 005, 016 and 019 routed correctly (`tests/test_llm_mock.py`). The policy layer, not model choice, drives the outcome.

## What I chose not to build
- **No actions or integrations** (sending replies, CRM, order or booking lookups). Out of scope, and the routing decision is what's being assessed.
- **No queue, database or web service.** A command-line tool over a file shows the logic. At 10k a day (about 7 a minute) the same `triage_one()` sits behind any queue worker.
- **No knowledge base answers.** "Automation acts" means *a reply drawn from a curated source*, which is not included here.
- **No attachment parsing** (the file has none). Mentions of one are flagged instead.
- **No PII redaction before the LLM call.** This is the first thing I'd add before production, because free-tier providers may log prompts that contain phone numbers and health questions.

## Where this breaks
- **There is no customer or thread ID in the data.** Without one I can't link 007's phone number to its conversation, detect real repeat contact, or resolve 012 or 020. This is the biggest gap, and it's in the data, not the code.
- **Confidence is not calibrated.** The model reports 0.9–0.99 for almost everything, and my penalties and the 0.75 threshold are judgment calls, not fitted to labelled data.
- **The regex safety net is English-only.** A health question or injection in Hindi relies on the model alone.
- **The free tier isn't real infrastructure.** `gemma-4-31b-it:free` returned 429 (upstream shared pool) on nearly every call. On `nemotron-3.5-lightning:free`, the 50-requests-a-day cap ran out mid-run, and 4 messages were cut off at a 4k `max_tokens` because the model spent the whole budget reasoning. Those 5 messages went to rules-only and to a human, which is the intended behaviour, and the committed `results.json` shows it. Backoff, the fallback model, a circuit breaker, the cache and rules-only mode keep it running.
- **Reasoning tokens are where the cost goes.** During development, Lightning used a median of about 2,900 output tokens to produce a roughly 300-token label, and `effort=low` barely changed that. The tool now defaults to `qwen/qwen3.8-27b:free` on OpenRouter with reasoning at the lowest level each provider allows. The provider is pluggable (OpenAI, Claude, OpenRouter, Groq, Gemini, Ollama, any OpenAI-compatible URL), and when an endpoint rejects an optional parameter (JSON mode, `temperature`, effort, `max_tokens`), the client drops it and relies on the prompt plus a tolerant parser.
- **The order and booking ID formats are guessed** from 25 samples. Timestamps look like they're in server time (016 is in Toronto but stamped +05:30), so they shouldn't drive SLAs.

## Cost (measured on these runs)
Average tokens per message measured from the API's `usage` field, with paid OpenRouter prices checked 2026-10-02:

| Model | tokens in / out | $ / M in / out | **$ per 1,000 msgs** | 10k/day, per month |
|---|---|---|---|---|
| **qwen3.8-27b (the model used), reasoning off** | ~1,000 / ~330 (estimated) | 0.42 / 3.00 | **~$1.41** | ~$423 |
| nemotron-3.5-lightning (measured in dev) | 1,090 / 2,870 | 0.06 / 0.16 | $0.53 | ~$158 |
| nemotron-3-super | 950 / 710 | 0.08 / 0.45 | $0.39 | ~$117 |
| gemma-4-31b-it (non-reasoning) | 1,000 / 330 | 0.09 / 0.34 | **$0.20** | ~$60 |
| claude-haiku-4.5 (for comparison, estimated) | ~1,000 / ~330 | 1.00 / 5.00 | ~$2.65 | ~$795 |

The Qwen row is an estimate: its token counts come from the non-reasoning runs, because the free quota ran out before a live Qwen run. The tool prints the measured figure after every run. Qwen's $3/M output price makes it about 7× the cost of Gemma per message. Even so, at 10k a day the LLM bill (about $420 a month) is small next to the cost of human review. Assumptions: one call per message, plus about 10% for retries and repairs. 8% of messages (empty or contact-only) skip the LLM. Real emails are longer than these, so budget about 1.5× (**about $2.10 per 1,000 on Qwen**). The ~850-token system prompt is most of the input, and prompt caching would cut it further. **The LLM is not the real cost: human review is.** At a 20–40% review rate on normal traffic, that's 2–4k messages a day for people to handle, so the threshold is the lever that actually matters.

## With another day
1. Label 300–500 real messages, then set the threshold *per intent* against a precision target, and compare models on those labels (accuracy, cost, p95 latency).
2. Run in shadow mode: automation proposes, a human decides, and we measure agreement before turning anything on.
3. Add a customer and thread lookup, PII redaction, a knowledge base for each brand behind the auto-reply intents, and Hindi or Hinglish rule patterns.

## AI tools used
- **Claude Code (Claude Opus 5.5):** read the brief and data and proposed the plan (which I reviewed and changed: OpenRouter, Python, a command-line tool, and Qwen 3.8 27B as the only model). It wrote `triage.py`, the README and the first draft of this log, built the malformed inputs and the mock-API test, and ran the tool. I reviewed the outputs and edited this log.
- **At runtime:** whichever LLM is configured (see README, *Choosing the LLM*). The default is `qwen/qwen3.8-27b:free` on OpenRouter; Claude uses Anthropic's official SDK. During development I also tried Gemma 4 31B and Nemotron 3 Super / 3.5 Lightning (free tiers).
