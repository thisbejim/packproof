"""Deterministic offline checks for packed language-model training records."""

from .checks import audit_jsonl, audit_records
from .model import AuditReport, CheckConfig, Finding, Severity

__all__ = [
    "AuditReport",
    "CheckConfig",
    "Finding",
    "Severity",
    "audit_jsonl",
    "audit_records",
]

__version__ = "0.1.0"
