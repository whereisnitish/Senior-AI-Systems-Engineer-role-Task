"""Deterministic pre-checks: regex risk flags, entity extraction and the keyword fallback classifier."""
from __future__ import annotations

import re


INJECTION_PATTERNS = [
    r"ignore (all |any )?(the )?(previous|prior|above) (instructions|prompts?)",
    r"\byou are now\b", r"\b(administrator|admin|developer|god) mode\b", r"\bsystem (notice|prompt|override)\b",
    r"do not (escalate|tell|involve|forward)", r"\breply (only )?(with )?['\"]?confirmed\b",
    r"\bjailbreak\b", r"disregard (your|the) (rules|instructions)", r"</?(system|assistant)>",
]
PAYMENT_REDIRECT_PATTERNS = [r"\bbank details\b", r"\bprocess (the )?payment\b", r"\bwire (transfer|the)\b",
                             r"\b(new|updated|changed) (bank|account) (details|number)\b", r"\biban\b|\bswift\b"]
ATTACHMENT_PATTERNS = [r"\battach(ed|ment)\b", r"\benclosed\b", r"\bin the pdf\b", r"\bresume attached\b", r"\bsee (the )?pdf\b"]
CHARGEBACK_PATTERNS = [r"\b(raise|raising|report|dispute)\w* .{0,20}\bbank\b", r"\bchargeback\b", r"\bconsumer (court|forum)\b"]
LEGAL_PATTERNS = [r"\blawyer\b", r"\blegal action\b", r"\bsue\b"]
REPEAT_PATTERNS = [r"\b(second|third|fourth|fifth|\d+(st|nd|rd|th)) time\b", r"\bnobody (has )?(responded|replied)\b",
                   r"\bstill (no|waiting)\b", r"\bno (response|reply)\b"]
URGENT_PATTERNS = [r"\burgent\b", r"\bemergency\b", r"\bimmediately\b", r"\bstranded\b", r"\bcancell?ed\b.{0,40}\b(flight|connection)\b",
                   r"\b(flight|connection)\b.{0,40}\bcancell?ed\b"]
VULNERABLE_PATTERNS = [r"\belderly\b", r"\bwheelchair\b", r"\bover (7|8|9)0\b", r"\bdisab", r"\bpregnan", r"\binfant\b"]
HEALTH_PATTERNS = [r"\bsafe to take\b", r"\bmedication\b", r"\bside effects?\b", r"\bpregnan", r"\bdosage\b",
                   r"\bblood pressure\b", r"\ballerg", r"\binteract(ion)?s? with\b", r"\bdoctor\b"]
SELF_HARM_PATTERNS = [r"\bkill myself\b", r"\bsuicid", r"\bend my life\b", r"\bself[- ]harm\b"]

ORDER_ID_RE = re.compile(r"\b[A-Z]{2,4}-\d{4,8}\b")
BOOKING_REF_RE = re.compile(r"\b(?=[A-Z0-9]{6,8}\b)(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{6,8}\b")
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"(?<!\w)(\+?\d[\d\s().-]{7,}\d)(?!\w)")
AMOUNT_RE = re.compile(r"(?:\b(INR|Rs\.?|USD|CAD|EUR|AED)\s?|[₹$€])\s?(\d[\d,]*(?:\.\d+)?)|(\d[\d,]*(?:\.\d+)?)\s?(INR|rupees|USD|CAD)\b", re.I)


def matches_any(patterns, text):
    return any(re.search(p, text, re.I) for p in patterns)


def digits_only(s):
    return re.sub(r"\D", "", s)


def regex_entities(text: str) -> dict:
    phones = []
    for m in PHONE_RE.findall(text):
        d = digits_only(m)
        if 10 <= len(d) <= 15:
            phones.append(m.strip())
    amounts = []
    for m in AMOUNT_RE.finditer(text):
        cur = (m.group(1) or m.group(4) or "").upper().replace("RS.", "INR").replace("RS", "INR").replace("RUPEES", "INR")
        if not cur:
            cur = {"₹": "INR", "$": "USD?", "€": "EUR"}.get(m.group(0).strip()[0], "")
        val = (m.group(2) or m.group(3) or "").replace(",", "")
        try:
            amounts.append({"value": float(val), "currency": cur or None})
        except ValueError:
            pass
    order_ids = sorted(set(ORDER_ID_RE.findall(text)))
    booking_refs = sorted(set(r for r in BOOKING_REF_RE.findall(text)
                              if r not in order_ids and not r.isdigit() and not any(r in o for o in order_ids)))
    return {
        "order_ids": order_ids,
        "booking_refs": booking_refs,
        "phone_numbers": phones,
        "emails": sorted(set(EMAIL_RE.findall(text))),
        "amounts": amounts,
    }


def rule_flags(text: str) -> set:
    flags = set()
    checks = [("prompt_injection", INJECTION_PATTERNS), ("payment_redirect_risk", PAYMENT_REDIRECT_PATTERNS),
              ("missing_attachment", ATTACHMENT_PATTERNS), ("chargeback_threat", CHARGEBACK_PATTERNS),
              ("legal_threat", LEGAL_PATTERNS), ("repeat_contact", REPEAT_PATTERNS),
              ("urgent_safety", URGENT_PATTERNS), ("vulnerable_customer", VULNERABLE_PATTERNS),
              ("health_safety", HEALTH_PATTERNS), ("self_harm", SELF_HARM_PATTERNS)]
    for name, pats in checks:
        if matches_any(pats, text):
            flags.add(name)
    # "urgent" alone in a sales email is not a safety issue; require a vulnerable person or disruption wording too
    if "urgent_safety" in flags and not (matches_any(URGENT_PATTERNS[3:], text) or "vulnerable_customer" in flags):
        flags.discard("urgent_safety")
    # payment-redirect only matters if it is about us paying someone
    if "payment_redirect_risk" in flags and not re.search(r"invoice|pay(ment)?|bank", text, re.I):
        flags.discard("payment_redirect_risk")
    return flags


def is_contact_only(text: str) -> bool:
    stripped = re.sub(r"[\s+().-]", "", text)
    return stripped.isdigit() and 8 <= len(stripped) <= 15 or bool(re.fullmatch(EMAIL_RE.pattern, text.strip()))


# --------------------------------------------------------------------------------------
# Rules-only fallback classifier (used when the LLM is unavailable). Deliberately crude:
# its confidence is capped so everything it touches goes to a human, pre-labelled.
# --------------------------------------------------------------------------------------
KEYWORD_INTENTS = [
    ("suspicious_instruction", INJECTION_PATTERNS),
    ("travel_disruption", [r"cancell?ed.{0,40}(flight|connection)|(flight|connection).{0,40}cancell?ed|stranded"]),
    ("health_safety_question", [r"safe to take|medication|side effect"]),
    ("billing_dispute", [r"charged twice|double charge|duplicate charge|charged two"]),
    ("vendor_invoice", [r"\binvoice\b"]),
    ("job_application", [r"resume|cv\b|openings?|vacanc|hiring"]),
    ("wholesale_inquiry", [r"wholesale|bulk pricing|distributor"]),
    ("cancel_subscription", [r"cancel (my )?subscription"]),
    ("marketing_unsubscribe", [r"^\s*unsubscribe\s*$"]),
    ("refund_request", [r"\brefund\b"]),
    ("return_or_exchange", [r"wrong item|damaged|broken|got the \w+ instead"]),
    ("order_status", [r"tracking|where is my order|hasn'?t (arrived|moved)"]),
    ("booking_change", [r"change my (return|flight|booking)|change fee"]),
    ("quote_acceptance", [r"quote you sent|how do we pay|next step"]),
    ("travel_inquiry", [r"fares?|flights? to|itinerary"]),
    ("appointment_reschedule", [r"shift it|move my .{0,20}appointment|reschedule"]),
    ("appointment_cancel", [r"something came up|can'?t make it|cancel (my )?appointment"]),
    ("business_hours_inquiry", [r"\bopen on\b|opening hours|what time do you"]),
    ("service_price_inquiry", [r"how much|\brate\b|price"]),
    ("appointment_book", [r"\bbook(ing)?\b"]),
    ("positive_feedback", [r"thank you|obsessed|amazing job|love (it|my)"]),
]


def keyword_intents(text: str) -> list:
    found = []
    for name, pats in KEYWORD_INTENTS:
        if matches_any(pats, text) and name not in found:
            found.append(name)
    return found[:3] or ["unclear"]
