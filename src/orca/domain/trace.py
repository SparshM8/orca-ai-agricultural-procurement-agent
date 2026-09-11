"""Observational Agent Trace and Decision Audit domain models."""

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

SENSITIVE_KEY_SUBSTRINGS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "passwd",
    "secret",
    "access_token",
    "refresh_token",
    "token",
    "credentials",
    "credential",
    "client_secret",
    "private_key",
)

# Regex patterns for inline credential detection in arbitrary text
CREDENTIAL_PATTERNS = [
    # Gemini / Google style API keys
    (re.compile(r"(?i)AQ\.[A-Za-z0-9_\-]{20,}"), "[REDACTED_API_KEY]"),
    (re.compile(r"(?i)AIza[0-9A-Za-z\-_]{35}"), "[REDACTED_API_KEY]"),
    # Bearer tokens
    (re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.]{10,}"), "Bearer [REDACTED_TOKEN]"),
    # Key=value or key: value credential strings
    (
        re.compile(
            r"(?i)\b(api_key|apikey|secret|token|password|bearer|auth|credentials?)\s*[:=]\s*['\"]?([A-Za-z0-9_\-\.]{6,})['\"]?"
        ),
        r"\1=[REDACTED]",
    ),
]


def sanitize_text(text: Optional[str], max_len: int = 300) -> str:
    """Sanitize arbitrary text string: redacting credentials and truncating to max_len."""
    if not text:
        return ""

    sanitized = str(text)
    for pattern, replacement in CREDENTIAL_PATTERNS:
        sanitized = pattern.sub(replacement, sanitized)

    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len] + "... [truncated]"

    return sanitized


def sanitize_payload(obj: Any, max_string_len: int = 300) -> Any:
    """Recursively sanitize dicts, lists, and primitives against sensitive keys and credentials."""
    if isinstance(obj, dict):
        sanitized_dict = {}
        for k, v in obj.items():
            k_lower = str(k).lower()
            if any(sub in k_lower for sub in SENSITIVE_KEY_SUBSTRINGS):
                sanitized_dict[k] = "[REDACTED]"
            else:
                sanitized_dict[k] = sanitize_payload(v, max_string_len=max_string_len)
        return sanitized_dict
    elif isinstance(obj, (list, tuple, set)):
        return [sanitize_payload(item, max_string_len=max_string_len) for item in obj]
    elif isinstance(obj, str):
        return sanitize_text(obj, max_len=max_string_len)
    elif isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    else:
        return sanitize_text(str(obj), max_len=max_string_len)


class AgentTrace(BaseModel):
    """Observational record capturing the decision lineage for an inbound conversational turn.
    
    CRITICAL: This model is purely observational. It has zero authority over
    order, payment, pricing, collection, or dialogue state transitions.
    """

    trace_id: str = Field(default_factory=lambda: f"trc_{uuid.uuid4().hex[:12]}")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    conversation_id: str
    sender_id: str
    message_id: Optional[str] = None
    input_text: str = ""  # Sanitized and bounded preview
    detected_intent: str = "UNKNOWN"
    confidence: float = 1.0
    extracted_entities: Dict[str, Any] = Field(default_factory=dict)
    classifier_used: str = "RuleBasedIntentClassifier"
    fallback_occurred: bool = False
    fallback_reason: Optional[str] = None
    selected_tool: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    tool_success: Optional[bool] = None
    tool_error: Optional[str] = None
    state_before: str = "OFFER_RECEIVED"
    state_after: str = "OFFER_RECEIVED"
    response_path: str = "deterministic"
    error: Optional[str] = None

    @classmethod
    def create_sanitized(
        cls,
        conversation_id: str,
        sender_id: str,
        state_before: str,
        input_text: str,
        trace_id: Optional[str] = None,
        message_id: Optional[str] = None,
        detected_intent: str = "UNKNOWN",
        confidence: float = 1.0,
        extracted_entities: Optional[Dict[str, Any]] = None,
        classifier_used: str = "RuleBasedIntentClassifier",
        fallback_occurred: bool = False,
        fallback_reason: Optional[str] = None,
        selected_tool: Optional[str] = None,
        tool_arguments: Optional[Dict[str, Any]] = None,
        tool_success: Optional[bool] = None,
        tool_error: Optional[str] = None,
        state_after: Optional[str] = None,
        response_path: str = "deterministic",
        error: Optional[str] = None,
    ) -> "AgentTrace":
        """Factory method applying thorough sanitization to all input and diagnostic fields."""
        return cls(
            trace_id=trace_id or f"trc_{uuid.uuid4().hex[:12]}",
            conversation_id=conversation_id,
            sender_id=sender_id,
            message_id=message_id,
            input_text=sanitize_text(input_text, max_len=250),
            detected_intent=detected_intent,
            confidence=round(confidence, 4),
            extracted_entities=sanitize_payload(extracted_entities or {}),
            classifier_used=classifier_used,
            fallback_occurred=fallback_occurred,
            fallback_reason=sanitize_text(fallback_reason, max_len=200) if fallback_reason else None,
            selected_tool=selected_tool,
            tool_arguments=sanitize_payload(tool_arguments) if tool_arguments is not None else None,
            tool_success=tool_success,
            tool_error=sanitize_text(tool_error, max_len=200) if tool_error else None,
            state_before=state_before,
            state_after=state_after or state_before,
            response_path=response_path,
            error=sanitize_text(error, max_len=200) if error else None,
        )
