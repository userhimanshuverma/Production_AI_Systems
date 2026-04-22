"""
Structured JSON logger.
Every log entry includes trace_id, step, timestamp.
"""
import json
import logging
import time
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
_logger = logging.getLogger("rag_system")


def log(trace_id: str, step: str, data: dict[str, Any]) -> None:
    entry = {
        "trace_id": trace_id,
        "step": step,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **data,
    }
    _logger.info(json.dumps(entry))
