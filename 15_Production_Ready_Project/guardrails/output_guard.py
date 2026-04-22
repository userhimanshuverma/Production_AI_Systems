"""
Output guardrails: PII redaction, length check, format validation.
"""
import re

MIN_RESPONSE_LENGTH = 10
MAX_RESPONSE_LENGTH = 2000

PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]"),
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL REDACTED]"),
    (r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b", "[CARD REDACTED]"),
    (r"\b\+?1?\s?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", "[PHONE REDACTED]"),
]


def redact_pii(text: str) -> tuple[str, bool]:
    """Returns (redacted_text, was_redacted)."""
    redacted = text
    changed = False
    for pattern, replacement in PII_PATTERNS:
        new_text = re.sub(pattern, replacement, redacted)
        if new_text != redacted:
            changed = True
        redacted = new_text
    return redacted, changed


def validate(response: str) -> tuple[bool, str]:
    """Returns (is_valid, issue_type)."""
    if not response or len(response.strip()) < MIN_RESPONSE_LENGTH:
        return False, "response_too_short"
    if len(response) > MAX_RESPONSE_LENGTH:
        return False, "response_too_long"
    return True, ""
