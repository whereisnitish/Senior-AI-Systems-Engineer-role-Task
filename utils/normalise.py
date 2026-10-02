"""Turn any element of the input file into a safe record (never raises)."""
from __future__ import annotations

import unicodedata
from datetime import datetime, timezone

from utils.config import MAX_TEXT_CHARS
from utils.taxonomy import KNOWN_BRANDS, KNOWN_CHANNELS


def normalise(raw, index: int, seen_ids: set) -> dict:
    flags = set()
    if not isinstance(raw, dict):
        raw = {"text": raw if isinstance(raw, str) else None}
        flags.add("malformed_record")

    mid = raw.get("id")
    mid = str(mid).strip() if mid not in (None, "") else f"ROW-{index + 1:05d}"
    if mid in seen_ids:
        flags.add("duplicate_id")
        mid = f"{mid}#dup{index + 1}"
    seen_ids.add(mid)

    brand = str(raw.get("brand") or "").strip().lower() or None
    if brand not in KNOWN_BRANDS:
        flags.add("unknown_brand")
    channel = str(raw.get("channel") or "").strip().lower() or None
    if channel not in KNOWN_CHANNELS:
        flags.add("unknown_channel")

    received_raw = raw.get("received_at")
    received_utc = None
    try:
        dt = datetime.fromisoformat(str(received_raw).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            flags.add("invalid_timestamp")          # naive timestamp: timezone unknown
        else:
            received_utc = dt.astimezone(timezone.utc).isoformat()
    except (TypeError, ValueError):
        flags.add("invalid_timestamp")

    text = raw.get("text")
    if isinstance(text, (list, tuple)):
        text = " ".join(str(t) for t in text if t is not None)
    elif text is not None and not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize("NFKC", text or "")
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C").strip()
    if len(text) > MAX_TEXT_CHARS:
        text = text[:MAX_TEXT_CHARS]
        flags.add("truncated")

    return {"id": mid, "brand": brand, "channel": channel, "received_at": received_raw,
            "received_at_utc": received_utc, "text": text,
            "has_attachments_field": bool(raw.get("attachments")), "input_flags": flags}
