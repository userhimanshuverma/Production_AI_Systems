"""
Input guardrails: scope check, injection detection, length limit.
All checks are deterministic — fast and cheap.
"""
import re

MAX_INPUT_LENGTH = 600

INJECTION_PATTERNS = [
    r"ignore\s+(your\s+|all\s+|previous\s+)?instructions",
    r"forget\s+(your\s+|all\s+|previous\s+)?instructions",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a\s+|an\s+)?",
    r"reveal\s+(your\s+|the\s+)?system\s+prompt",
    r"what\s+(are|were)\s+your\s+instructions",
    r"disregard\s+",
    r"override\s+",
]

# Define what topics the system handles — extend for your domain
IN_SCOPE_KEYWORDS = [
    "policy", "refund", "return", "shipping", "order", "product",
    "price", "pricing", "support", "account", "payment", "cancel",
    "document", "process", "procedure", "how", "what", "when", "where",
    "expense", "reimbursement", "access", "database", "deploy",
]


def check_injection(text: str) -> bool:
    """Returns True if injection pattern detected."""
    lower = text.lower()
    return any(re.search(p, lower) for p in INJECTION_PATTERNS)


def check_scope(text: str) -> bool:
    """Returns True if query appears in-scope."""
    lower = text.lower()
    return any(kw in lower for kw in IN_SCOPE_KEYWORDS)


def validate(query: str) -> tuple[bool, str]:
    """
    Returns (is_valid, rejection_reason).
    Empty reason string means valid.
    """
    if not query or not query.strip():
        return False, "empty_query"
    if len(query) > MAX_INPUT_LENGTH:
        return False, "input_too_long"
    if check_injection(query):
        return False, "injection_detected"
    if not check_scope(query):
        return False, "out_of_scope"
    return True, ""
