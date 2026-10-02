"""Parse and validate model output: extract the JSON, coerce it to the schema, drop invented entities."""
from __future__ import annotations

import json
import re

from utils.rules import digits_only
from utils.taxonomy import INTENTS, LLM_ALLOWED_FLAGS


def parse_json_object(text: str):
    """Pull a JSON object out of model output that may be fenced or wrapped in prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def str_list(v, limit=20):
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if x not in (None, "") and not isinstance(x, (dict, list))][:limit]


def validate_llm(obj: dict, text: str) -> tuple[dict, list]:
    """Coerce model output into the expected shape. Returns (clean, issues)."""
    issues = []
    intents = []
    raw_intents = obj.get("intents")
    if isinstance(raw_intents, dict):
        raw_intents = [raw_intents]
    for it in raw_intents if isinstance(raw_intents, list) else []:
        if isinstance(it, str):
            it = {"type": it}
        if not isinstance(it, dict):
            continue
        t = str(it.get("type") or "").strip().lower().replace(" ", "_")
        if t not in INTENTS:
            issues.append(f"unknown intent type '{t}' -> other")
            t = "other"
        details = it.get("details") if isinstance(it.get("details"), dict) else {}
        intents.append({"type": t, "summary": str(it.get("summary") or "")[:300], "details": details})
    if not intents:
        issues.append("model returned no intents")

    ents = obj.get("entities") if isinstance(obj.get("entities"), dict) else {}
    norm_text = re.sub(r"[\s-]", "", text).upper()
    text_digits = digits_only(text)

    def grounded(values, key):
        kept = []
        for v in str_list(values):
            probe = re.sub(r"[\s-]", "", v).upper()
            ok = probe in norm_text if key != "phone_numbers" else (digits_only(v) and digits_only(v) in text_digits)
            if ok:
                kept.append(v)
            else:
                issues.append(f"dropped ungrounded {key[:-1]} '{v}'")
        return kept

    amounts = []
    for a in ents.get("amounts") if isinstance(ents.get("amounts"), list) else []:
        if isinstance(a, dict):
            try:
                val = float(str(a.get("value")).replace(",", ""))
            except (TypeError, ValueError):
                continue
            if val == 0:          # template placeholder echoed back, not an extraction
                continue
            if digits_only(f"{val:.0f}") in text_digits or digits_only(str(a.get("value"))) in text_digits:
                amounts.append({"value": val, "currency": a.get("currency")})
            else:
                issues.append(f"dropped ungrounded amount '{a.get('value')}'")

    clean = {
        "language": str(obj.get("language") or "unknown")[:20],
        "intents": intents[:6],
        "entities": {
            "order_ids": grounded(ents.get("order_ids"), "order_ids"),
            "booking_refs": grounded(ents.get("booking_refs"), "booking_refs"),
            "phone_numbers": grounded(ents.get("phone_numbers"), "phone_numbers"),
            "emails": grounded(ents.get("emails"), "emails"),
            "amounts": amounts,
            "dates_mentioned": str_list(ents.get("dates_mentioned")),
            "products_or_services": str_list(ents.get("products_or_services")),
            "people_named": str_list(ents.get("people_named")),
            "airline": str(ents["airline"]) if isinstance(ents.get("airline"), str) and ents["airline"] else None,
            "locations": str_list(ents.get("locations")),
        },
        "sentiment": obj.get("sentiment") if obj.get("sentiment") in ("positive", "neutral", "negative", "angry") else "neutral",
        "risk_flags": {f for f in str_list(obj.get("risk_flags")) if f in LLM_ALLOWED_FLAGS},
        "ambiguity": obj.get("ambiguity") if obj.get("ambiguity") in ("none", "low", "high") else "low",
        "ambiguity_note": str(obj.get("ambiguity_note") or "")[:300],
        "needs_prior_context": obj.get("needs_prior_context") is True,
        "confidence": clamp(obj.get("confidence"), default=0.5),
        "note_for_agent": str(obj.get("note_for_agent") or "")[:400],
    }
    return clean, issues


def clamp(v, default=0.5):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f:  # NaN
        return default
    return max(0.0, min(1.0, f / 100 if 1 < f <= 100 else f))


def merge_entities(llm_ents: dict, rx: dict) -> dict:
    out = dict(llm_ents)
    for k in ("order_ids", "booking_refs", "phone_numbers", "emails"):
        seen = {re.sub(r"[\s-]", "", x).upper() for x in out.get(k, [])}
        for v in rx.get(k, []):
            if re.sub(r"[\s-]", "", v).upper() not in seen:
                out.setdefault(k, []).append(v)
    if not out.get("amounts") and rx.get("amounts"):
        out["amounts"] = rx["amounts"]
    return out
