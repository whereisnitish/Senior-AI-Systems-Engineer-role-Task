"""Per-message pipeline (pre-checks -> LLM -> validate -> policy) and the concurrent batch runner."""
from __future__ import annotations

import concurrent.futures as cf

from services.llm_client import LLMError, OpenRouterClient
from services.policy import apply_policy
from utils.cache import Cache
from utils.llm_output import merge_entities, parse_json_object, validate_llm
from utils.prompts import PROMPT_VERSION, REPAIR_PROMPT, SYSTEM_PROMPT, build_user_message
from utils.rules import is_contact_only, keyword_intents, regex_entities, rule_flags


def triage_one(rec: dict, client: OpenRouterClient | None, cache: Cache) -> dict:
    text = rec["text"]
    rx = regex_entities(text) if text else {}
    rec["input_flags"] |= rule_flags(text) if text else {"empty_message"}
    base = {k: rec[k] for k in ("id", "brand", "channel", "received_at", "received_at_utc")}
    base["text_preview"] = text[:160]
    usage, model_used, issues, mode = {}, None, [], "llm"

    if not text:
        analysis = {"intents": [{"type": "unclear", "summary": "Empty message (no text; possibly media-only)",
                                 "details": {}}], "entities": {}, "confidence": 0.0, "ambiguity": "high",
                    "language": None, "sentiment": "neutral", "risk_flags": set(),
                    "note_for_agent": "No text content. Check the channel for an image/voice note."}
        mode = "precheck"
    elif is_contact_only(text):
        analysis = {"intents": [{"type": "contact_details_only", "summary": "Sender only shared contact details",
                                 "details": {}}], "entities": rx, "confidence": 0.9, "ambiguity": "high",
                    "needs_prior_context": True, "language": None, "sentiment": "neutral", "risk_flags": set(),
                    "note_for_agent": "Likely a reply to an earlier request for a callback number."}
        mode = "precheck"
    else:
        analysis = None
        if client:
            user_msg = build_user_message(rec)
            ckey = cache.key(PROMPT_VERSION, client.model, client.reasoning, rec["brand"] or "", user_msg)
            cached = cache.get(ckey)
            try:
                if cached:
                    obj, usage, model_used = cached["obj"], cached.get("usage", {}), cached.get("model")
                else:
                    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
                    content, usage, model_used = client.chat(msgs)
                    obj = parse_json_object(content)
                    if obj is None and usage.get("finish_reason") == "length":
                        issues.append("model output cut off at max_tokens; not retried")
                    elif obj is None:   # one repair attempt
                        issues.append("json_repaired")
                        msgs += [{"role": "assistant", "content": content[:2000]},
                                 {"role": "user", "content": REPAIR_PROMPT}]
                        content, usage2, model_used = client.chat(msgs)
                        usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in ("prompt_tokens", "completion_tokens")}
                        obj = parse_json_object(content)
                    if obj is not None:
                        cache.put(ckey, {"obj": obj, "usage": usage, "model": model_used})
                if obj is not None:
                    analysis, v_issues = validate_llm(obj, text)
                    issues += v_issues
                else:
                    if "json_repaired" in issues:
                        issues.append("model output unparseable after repair retry")
            except LLMError as e:
                issues.append(f"llm_error: {e}")
        if analysis is None:
            mode = "rules_only"
            rec["input_flags"].add("llm_unavailable")
            analysis = {"intents": [{"type": t, "summary": "(keyword match)", "details": {}} for t in keyword_intents(text)],
                        "entities": {}, "confidence": 0.4, "ambiguity": "low", "language": None,
                        "sentiment": "neutral", "risk_flags": set(), "note_for_agent": ""}

    entities = merge_entities(analysis.get("entities") or {}, rx)
    decision = apply_policy(rec, analysis, mode, issues)
    return {
        **base,
        "language": analysis.get("language"),
        "sentiment": analysis.get("sentiment"),
        **decision,
        "entities": entities,
        "ambiguity_note": analysis.get("ambiguity_note") or None,
        "note_for_agent": analysis.get("note_for_agent") or None,
        "processing": {"mode": mode, "model": model_used, "issues": issues,
                       "tokens_in": usage.get("prompt_tokens", 0), "tokens_out": usage.get("completion_tokens", 0)},
    }


def safe_triage(rec, client, cache):
    try:
        return triage_one(rec, client, cache)
    except Exception as e:   # last line of defence: one bad message must never stop the batch
        return {"id": rec.get("id"), "brand": rec.get("brand"), "channel": rec.get("channel"),
                "intents": [], "priority": "high", "risk_flags": ["processing_error"], "confidence": 0.0,
                "needs_human": True, "human_reasons": [f"processing error: {type(e).__name__}: {e}"],
                "primary_queue": "general_triage", "processing": {"mode": "error", "issues": [repr(e)]}}


def run_batch(records: list, client: OpenRouterClient | None, cache: Cache, workers: int = 2,
              on_result=None) -> list:
    """Triage every record concurrently; results keep input order. on_result(done, total, result) is optional."""
    results = [None] * len(records)
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(safe_triage, rec, client, cache): i for i, rec in enumerate(records)}
        for done, fut in enumerate(cf.as_completed(futs), 1):
            i = futs[fut]
            results[i] = fut.result()
            if on_result:
                on_result(done, len(records), results[i])
    return results
