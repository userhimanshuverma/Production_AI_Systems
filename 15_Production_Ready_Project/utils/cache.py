"""
Simple in-memory response cache with TTL.
Drop-in replaceable with Redis: swap get/set implementations only.
"""
import hashlib
import time
from typing import Optional

_store: dict[str, tuple[str, float]] = {}  # key -> (value, expires_at)


def _key(text: str) -> str:
    return hashlib.md5(text.strip().lower().encode()).hexdigest()


def get(query: str) -> Optional[str]:
    k = _key(query)
    if k in _store:
        value, expires_at = _store[k]
        if time.time() < expires_at:
            return value
        del _store[k]
    return None


def set(query: str, value: str, ttl: int = 3600) -> None:
    _store[_key(query)] = (value, time.time() + ttl)


def invalidate(query: str) -> None:
    _store.pop(_key(query), None)
