"""Optional request-parameter tracing; it reads no response bodies and does not alter
retry or network behavior."""

import json
import re
from collections.abc import Mapping
from itertools import islice
from time import perf_counter
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from loguru import logger
from pydantic import BaseModel

from app.utils.logging_utils import get_logging_options


_REDACTED = "[REDACTED]"
_MAX_STRING = 1000
_MAX_PAYLOAD = 12000
# Credential field names are always redacted. Device-login session identifiers
# are credentials too: Kimi uses ``device_code`` and Codex uses
# ``device_auth_id``, and both must stay out of the logs.
_SECRET_KEY = re.compile(
    r"authorization|cookie|password|passwd|secret|credential|signature|"
    r"apikey|accesskey|privatekey|clientkey|devicecode|deviceauthid|usercode|"
    r"authorizationcode|(?:access|refresh|id|auth|api|bearer)token"
)
_TOKEN = re.compile(
    r"\b(?:sk|rk|sess|ghp|github_pat)[-_][A-Za-z0-9_-]{8,}|"
    r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"
)
_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_MEDIA_KEYS = frozenset(
    {
        "audio",
        "audiodata",
        "audiobase64",
        "base64",
        "b64json",
        "bytes",
        "file",
        "files",
        "filedata",
        "image",
        "imagedata",
        "imagebase64",
        "video",
    }
)


def _safe_url(value: str) -> str:
    """Redact every query parameter so vendor-specific signature fields cannot leak."""
    try:
        parts = urlsplit(value)
        netloc = parts.netloc.rsplit("@", 1)[-1]
        query = urlencode([(k, _REDACTED) for k, _ in parse_qsl(parts.query)])
        return _TOKEN.sub(
            _REDACTED, urlunsplit((parts.scheme, netloc, parts.path, query, ""))
        )[:_MAX_STRING]
    except ValueError:
        return "[invalid URL]"


def _safe_string(value: str) -> str:
    if value.lstrip().lower().startswith("data:"):
        return "[media omitted]"
    if len(value) >= 256 and re.fullmatch(r"[A-Za-z0-9+/=\r\n]+", value):
        return "[base64 omitted]"
    value = _TOKEN.sub(_REDACTED, value)
    value = _URL.sub(lambda match: _safe_url(match.group()), value)
    return value[:_MAX_STRING] + ("… [truncated]" if len(value) > _MAX_STRING else "")


def _safe_query_params(value):
    """Stay consistent with the URL query by redacting all values, while accepting
the requests parameter formats."""
    if isinstance(value, Mapping):
        return {_safe_string(str(key)): _REDACTED for key in islice(value, 30)}
    if isinstance(value, (list, tuple)):
        return [
            [_safe_string(str(pair[0])), _REDACTED]
            for pair in value[:30]
            if isinstance(pair, (list, tuple)) and len(pair) == 2
        ]
    return "[query values omitted]"


def sanitize_payload(value):
    """Limit depth, node count and text length; unknown objects are recorded only by
type, never by calling their repr."""
    remaining = 100

    def visit(item, depth=0):
        nonlocal remaining
        remaining -= 1
        if remaining < 0 or depth > 6:
            return "[truncated]"
        if isinstance(item, Mapping):
            result = {}
            for index, (key, child) in enumerate(item.items()):
                if index >= 30 or remaining <= 0:
                    result["…"] = "[truncated]"
                    break
                key = str(key)
                normalized = re.sub(r"[^a-z0-9]", "", key.lower())
                if normalized in {
                    "key",
                    "token",
                    "headers",
                    "cookies",
                    "code",
                    "codeverifier",
                    "nonce",
                } or _SECRET_KEY.search(normalized):
                    result[_safe_string(key)] = _REDACTED
                elif normalized in {"params", "extraquery", "queryparams"}:
                    result[_safe_string(key)] = _safe_query_params(child)
                elif normalized in _MEDIA_KEYS:
                    result[_safe_string(key)] = "[media omitted]"
                else:
                    result[_safe_string(key)] = visit(child, depth + 1)
            return result
        if isinstance(item, (list, tuple)):
            return [visit(child, depth + 1) for child in item[:10]] + (
                ["[truncated]"] if len(item) > 10 else []
            )
        if isinstance(item, str):
            return _safe_string(item)
        if isinstance(item, BaseModel):
            return visit(item.model_dump(mode="json", exclude_none=True), depth + 1)
        if item is None or isinstance(item, (bool, int, float)):
            return item
        return f"[{type(item).__name__} omitted]"

    return visit(value)


def _emit(event: dict) -> None:
    # A diagnostic failure must not block a paid submission, pollute a business
    # exception or trigger an extra retry.
    try:
        logger.bind(request_trace=True).info(
            "request_trace {}", json.dumps(event, ensure_ascii=False)
        )
    except Exception:
        pass


def _trace_call(kind, method, endpoint, payload, send, *args, **kwargs):
    if not get_logging_options()[1]:
        return send(*args, **kwargs)
    trace_id = uuid4().hex[:12]
    try:
        parameters = sanitize_payload(payload)
        summary = json.dumps(parameters, ensure_ascii=False)
        _emit(
            {
                "event": "request",
                "id": trace_id,
                "kind": kind,
                "method": method,
                "endpoint": _safe_url(str(endpoint)),
                "parameters": parameters
                if len(summary) <= _MAX_PAYLOAD
                else summary[:_MAX_PAYLOAD],
                "truncated": len(summary) > _MAX_PAYLOAD,
            }
        )
    except Exception:
        pass
    started = perf_counter()
    try:
        response = send(*args, **kwargs)
    except Exception as exc:
        _emit(
            {
                "event": "error",
                "id": trace_id,
                "error_type": type(exc).__name__,
                "elapsed_ms": round((perf_counter() - started) * 1000, 1),
            }
        )
        raise
    # Do not touch .content/.text/.json(): that would eagerly consume SSE
    # streams, audio and large file downloads.
    try:
        status = getattr(response, "status_code", None)
    except Exception:
        status = None
    _emit(
        {
            "event": "response",
            "id": trace_id,
            "status": status if isinstance(status, int) else None,
            "elapsed_ms": round((perf_counter() - started) * 1000, 1),
        }
    )
    return response


def trace_http_call(method, send, url, *args, **kwargs):
    """Wrap requests issued explicitly by the project, leaving arguments and the
returned object unchanged."""
    payload = {
        key: kwargs[key] for key in ("params", "json", "data", "files") if key in kwargs
    }
    # Raw form/binary bodies cannot be reliably redacted field by field, so only
    # structured parameters may be inspected.
    if "data" in payload and not isinstance(payload["data"], Mapping):
        payload["data"] = "[raw body omitted]"
    if method.upper() == "REQUEST":
        # Session.request takes the method as its first positional argument and
        # the URL as the second.
        method, url, args = url, args[0], args[1:]
        return _trace_call(
            "http", method, url, payload, send, method, url, *args, **kwargs
        )
    return _trace_call("http", method, url, payload, send, url, *args, **kwargs)


def trace_sdk_call(provider, endpoint, send, **kwargs):
    """Record SDK call arguments; this does not claim to cover internal SDK retries or
the final wire JSON."""
    return _trace_call("sdk", provider, endpoint, kwargs, send, **kwargs)
