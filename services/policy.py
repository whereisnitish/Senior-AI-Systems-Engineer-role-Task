"""Business rules: route, priority, confidence and the human-in-the-loop decision. The model never overrides these."""
from __future__ import annotations

from utils.config import AUTO_CONFIDENCE_THRESHOLD
from utils.taxonomy import BRAND_INTENTS, HARD_FLAGS, INTENTS, KNOWN_BRANDS, PRIORITY_RANK, SHARED_INTENTS


def apply_policy(rec: dict, analysis: dict, mode: str, validation_issues: list) -> dict:
    """Deterministic: decide route, priority, confidence and needs_human."""
    brand = rec["brand"]
    flags = set(rec["input_flags"]) | set(analysis.get("risk_flags", set()))
    intents = analysis["intents"] or [{"type": "unclear", "summary": "", "details": {}}]

    if analysis.get("needs_prior_context"):
        flags.add("missing_context")
    if analysis.get("ambiguity") == "high":
        flags.add("ambiguous")
    if len(intents) > 1:
        flags.add("multi_intent")
    if analysis.get("sentiment") in ("negative", "angry"):
        flags.add("negative_sentiment")
    if any("ungrounded" in i for i in validation_issues):
        flags.add("unverified_entity")
    if "missing_attachment" in flags and rec["has_attachments_field"]:
        flags.discard("missing_attachment")
    if "prompt_injection" in flags and not any(i["type"] == "suspicious_instruction" for i in intents):
        intents = [{"type": "suspicious_instruction", "summary": "Message contains instructions aimed at the system",
                    "details": {}}] + intents
    # health questions must never be auto-answered, even if the model called it product_question
    if "health_safety" in flags:
        for i in intents:
            if i["type"] == "product_question":
                i["type"] = "health_safety_question"

    allowed = BRAND_INTENTS.get(brand, set()) | SHARED_INTENTS
    routed = []
    for i in intents:
        pol = INTENTS[i["type"]]
        mismatch = brand in KNOWN_BRANDS and i["type"] not in allowed
        if mismatch:
            flags.add("brand_intent_mismatch")
        routed.append({**i, "handler": f"{brand or 'unknown'}/{pol['queue']}", "next_action": pol["action"],
                       "automation_eligible": pol["auto"] and not mismatch, "priority": pol["priority"]})

    # ---- confidence: start from the model's self-report, subtract for things we can observe
    conf = analysis.get("confidence", 0.5)
    reasons = [f"model self-reported {conf:.2f}"]

    def penalise(amount, why):
        nonlocal conf
        conf -= amount
        reasons.append(f"-{amount:.2f} {why}")

    if analysis.get("ambiguity") == "high":
        penalise(0.20, "high ambiguity")
    elif analysis.get("ambiguity") == "low":
        penalise(0.05, "some ambiguity")
    if len(intents) > 1:
        penalise(0.05 * (len(intents) - 1), f"{len(intents)} intents")
    if any(i["type"] in ("other", "unclear") for i in intents):
        penalise(0.20, "intent unclear/other")
    if "unverified_entity" in flags:
        penalise(0.20, "model produced entities not in text")
    if "brand_intent_mismatch" in flags:
        penalise(0.15, "intent does not fit brand")
    if "missing_context" in flags:
        penalise(0.20, "depends on context we don't have")
    if any(x.startswith("unknown intent") for x in validation_issues):
        penalise(0.15, "model used unknown label")
    if "json_repaired" in validation_issues:
        penalise(0.10, "model output needed repair/retry")
    if "truncated" in flags:
        penalise(0.10, "text truncated")
    if mode == "rules_only":
        conf = min(conf, 0.40)
        reasons.append("capped at 0.40: rules-only fallback (LLM unavailable)")
    conf = round(max(0.0, min(1.0, conf)), 2)

    # ---- human-in-the-loop rule
    hard = sorted(flags & HARD_FLAGS)
    human_reasons = []
    if hard:
        human_reasons.append("risk flag(s): " + ", ".join(hard))
    not_auto = [r["type"] for r in routed if not r["automation_eligible"]]
    if not_auto:
        human_reasons.append("needs human action: " + ", ".join(not_auto))
    if conf < AUTO_CONFIDENCE_THRESHOLD:
        human_reasons.append(f"confidence {conf:.2f} < {AUTO_CONFIDENCE_THRESHOLD}")
    needs_human = bool(human_reasons)

    # ---- priority
    prio = max((r["priority"] for r in routed), key=PRIORITY_RANK.get)
    if flags & {"urgent_safety", "self_harm"}:
        prio = "urgent"
    elif flags & {"chargeback_threat", "legal_threat", "repeat_contact", "prompt_injection", "payment_redirect_risk",
                  "health_safety"} and PRIORITY_RANK[prio] < 2:
        prio = "high"

    # ---- primary queue: who picks this up first
    if "prompt_injection" in flags:
        primary = f"{brand or 'unknown'}/trust_safety"
    elif needs_human:
        candidates = [r for r in routed if not r["automation_eligible"]] or routed
        primary = max(candidates, key=lambda r: PRIORITY_RANK[r["priority"]])["handler"]
        if primary.endswith("/automation"):       # automation-only intent that failed the gate -> a person reviews
            primary = f"{brand or 'unknown'}/general_triage"
    else:
        primary = "automation"

    if "payment_redirect_risk" in flags and "prompt_injection" not in flags:
        primary = f"{brand or 'unknown'}/finance_ap"
    # Automation must never act on part of a message a human is already handling, or on anything flagged
    for r in routed:
        r["automation_will_act"] = r["automation_eligible"] and not hard and conf >= AUTO_CONFIDENCE_THRESHOLD

    return {
        "intents": routed,
        "priority": prio,
        "risk_flags": sorted(flags),
        "confidence": conf,
        "confidence_reasons": reasons,
        "needs_human": needs_human,
        "human_reasons": human_reasons,
        "primary_queue": primary,
    }
