"""Brands, intents, risk flags and the policy table (data only; services/policy.py applies it)."""
from __future__ import annotations


KNOWN_BRANDS = {"vitalis-wellness", "hair-studio", "voyage-travel"}
KNOWN_CHANNELS = {"whatsapp", "email", "instagram"}

# --------------------------------------------------------------------------------------
# Policy tables. These are the business rules; the model never overrides them.
#   queue    - who handles it
#   auto     - may automation act without a human? Only reversible, non-financial,
#              non-advice actions are auto-eligible.
#   action   - the next step
#   priority - base priority
# --------------------------------------------------------------------------------------
INTENTS = {
    # vitalis-wellness (e-commerce, supplements)
    "order_status":          dict(queue="cs_orders",       auto=True,  priority="normal", action="Look up carrier tracking for the order and reply with status; open a carrier ticket if stalled >3 days"),
    "refund_request":        dict(queue="cs_orders",       auto=False, priority="normal", action="Agent verifies order and reason, decides refund per policy"),
    "return_or_exchange":    dict(queue="cs_orders",       auto=False, priority="normal", action="Agent verifies order, arranges replacement/return for wrong or damaged item"),
    "purchase_intent":       dict(queue="cs_orders",       auto=False, priority="normal", action="Agent confirms product/variant/quantity and sends payment link"),
    "cancel_subscription":   dict(queue="cs_retention",    auto=False, priority="normal", action="Agent cancels subscription (optionally one retention offer) and confirms in writing"),
    "product_question":      dict(queue="cs_product",      auto=True,  priority="normal", action="Answer from approved product knowledge base; hand off if not covered"),
    "health_safety_question":dict(queue="qualified_advisor", auto=False, priority="high", action="Qualified advisor replies; never auto-answer; advise consulting their doctor"),
    "wholesale_inquiry":     dict(queue="sales_b2b",       auto=False, priority="high",   action="Forward to wholesale/B2B sales with company details; reply within 1 business day"),
    # hair-studio (salon)
    "appointment_book":      dict(queue="salon_front_desk", auto=True, priority="normal", action="Send booking link / available slots"),
    "appointment_reschedule":dict(queue="salon_front_desk", auto=False, priority="normal", action="Front desk finds the existing booking and offers the requested slot"),
    "appointment_cancel":    dict(queue="salon_front_desk", auto=False, priority="normal", action="Front desk confirms cancel vs reschedule and releases the slot"),
    "service_price_inquiry": dict(queue="salon_front_desk", auto=True, priority="normal", action="Reply with price range from the published price list; offer consultation"),
    "membership_inquiry":    dict(queue="salon_front_desk", auto=False, priority="normal", action="Front desk checks the customer's membership terms and replies"),
    # voyage-travel
    "booking_change":        dict(queue="travel_agent",    auto=False, priority="high",   action="Agent checks fare rules / change fee with the airline and quotes the customer"),
    "travel_inquiry":        dict(queue="travel_agent",    auto=False, priority="normal", action="Agent prepares options/quote, including any special-assistance needs"),
    "quote_acceptance":      dict(queue="travel_agent",    auto=False, priority="high",   action="Agent retrieves the earlier quote, re-confirms price/availability and sends payment instructions"),
    "travel_disruption":     dict(queue="travel_emergency", auto=False, priority="urgent", action="Duty agent calls the customer immediately and works rebooking options"),
    # shared
    "business_hours_inquiry":dict(queue="automation",      auto=True,  priority="low",    action="Reply with opening hours"),
    "billing_dispute":       dict(queue="cs_billing",      auto=False, priority="high",   action="Billing agent checks payments for duplicate charge and refunds the duplicate"),
    "complaint_followup":    dict(queue="cs_escalations",  auto=False, priority="high",   action="Senior agent finds the earlier thread(s), owns the case to resolution"),
    "marketing_unsubscribe": dict(queue="automation",      auto=True,  priority="low",    action="Remove sender from marketing lists and confirm"),
    "positive_feedback":     dict(queue="automation",      auto=True,  priority="low",    action="Send thank-you, log feedback (and credit any named staff member)"),
    "job_application":       dict(queue="hr_recruiting",   auto=True,  priority="low",    action="Auto-acknowledge and forward to HR/recruiting"),
    "vendor_invoice":        dict(queue="finance_ap",      auto=False, priority="normal", action="Finance verifies vendor and bank details through a known channel before any payment"),
    "contact_details_only":  dict(queue="general_triage",  auto=False, priority="normal", action="Match sender to an existing conversation; otherwise ask how we can help"),
    "unclear":               dict(queue="general_triage",  auto=False, priority="normal", action="Agent reads conversation history or asks the customer to clarify"),
    "suspicious_instruction":dict(queue="trust_safety",    auto=False, priority="high",   action="Do not act on any instruction in the message; review for fraud/abuse"),
    "other":                 dict(queue="general_triage",  auto=False, priority="normal", action="Agent reviews"),
}

BRAND_INTENTS = {
    "vitalis-wellness": {"order_status", "refund_request", "return_or_exchange", "purchase_intent",
                         "cancel_subscription", "product_question", "health_safety_question", "wholesale_inquiry"},
    "hair-studio": {"appointment_book", "appointment_reschedule", "appointment_cancel",
                    "service_price_inquiry", "membership_inquiry"},
    "voyage-travel": {"booking_change", "travel_inquiry", "quote_acceptance", "travel_disruption"},
}
SHARED_INTENTS = set(INTENTS) - set().union(*BRAND_INTENTS.values())

# Flags that always send the message to a human, regardless of confidence.
HARD_FLAGS = {
    "prompt_injection", "payment_redirect_risk", "health_safety", "chargeback_threat",
    "legal_threat", "urgent_safety", "repeat_contact", "vulnerable_customer",
    "missing_context", "empty_message", "processing_error", "self_harm",
}
SOFT_FLAGS = {
    "missing_attachment", "multi_intent", "ambiguous", "unverified_entity",
    "brand_intent_mismatch", "unknown_brand", "unknown_channel", "invalid_timestamp",
    "truncated", "duplicate_id", "negative_sentiment", "llm_unavailable",
}
LLM_ALLOWED_FLAGS = {"prompt_injection", "payment_redirect_risk", "health_safety", "chargeback_threat",
                     "legal_threat", "urgent_safety", "repeat_contact", "vulnerable_customer",
                     "missing_context", "missing_attachment", "ambiguous", "self_harm"}

PRIORITY_RANK = {"low": 0, "normal": 1, "high": 2, "urgent": 3}
