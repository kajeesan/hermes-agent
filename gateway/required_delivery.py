"""Fail-closed gateway delivery for tool-owned disclosure contracts.

The trusted input to this module is the direct result observed when a tool
finishes successfully.  Model text, caller-supplied metadata, and filesystem
receipts never create a required-delivery record.

Product-specific renderers live in plugins.  Core only binds the exact tool
completion to the Hermes session/turn/tool-call identity, invokes the selected
renderer, guarantees trusted disclosure precedes optional model prose, and
binds the complete outbound payload to its authenticated gateway-event source
and digest.  The source destination remains private to core: it is never
provided to a renderer or copied into tool/model/public evidence.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import secrets
import threading
import time
from typing import Any, Optional


REQUIRED_DELIVERY_CONTRACT = "hermes-gateway-required-delivery-v1"
RENDERED_DELIVERY_CONTRACT = "hermes-gateway-rendered-delivery-v1"
FAIL_CLOSED_TEXT = (
    "Unable to deliver this governed response because its trusted disclosure "
    "could not be validated. No model-generated explanation was sent."
)

_RECORD_TTL_SECONDS = 600.0
_MAX_RECORDS = 512


@dataclass(frozen=True)
class RequiredToolCompletion:
    session_id: str
    turn_id: str
    tool_call_id: str
    tool_name: str
    api_request_id: str
    task_id: str
    result: dict[str, Any]
    renderer_id: str
    renderer_version: str
    recorded_at: float
    invalid_reason: str = ""


@dataclass(frozen=True)
class OutboundValidation:
    governed: bool
    valid: bool
    token: str = ""
    error: str = ""
    trusted_disclosure: str = ""
    optional_prose: str = ""


class PreparedGatewayPayload(str):
    """String payload carrying an opaque core-issued delivery-plan token."""

    def __new__(cls, value: str, token: str):
        instance = super().__new__(cls, value)
        instance.required_delivery_token = token
        return instance


@dataclass(frozen=True)
class _OutboundRecord:
    platform: str
    destination_id: str
    destination_topic_id: Optional[str]
    payload: str
    payload_sha256: str
    trusted_disclosure: str
    optional_prose: str
    token: str
    created_at: float


_lock = threading.RLock()
_completions: dict[tuple[str, str], dict[str, RequiredToolCompletion]] = {}
_completion_conflicts: set[tuple[str, str]] = set()
_outbound: dict[str, _OutboundRecord] = {}
_reserved_outbound: set[str] = set()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _completion_role(result: Any) -> tuple[str, str]:
    """Return the optional generic required-delivery sequencing marker."""

    if not isinstance(result, dict):
        return "", ""
    marker = result.get("delivery_contract")
    if not isinstance(marker, dict):
        return "", ""
    role = marker.get("completion_role")
    final_tool = marker.get("final_tool_name")
    return (
        role if isinstance(role, str) else "",
        final_tool if isinstance(final_tool, str) else "",
    )


def _canonical_destination_topic_id(
    platform: str, destination_topic_id: Any
) -> tuple[Optional[str], str]:
    """Canonicalize the bounded topic component of an outbound route.

    Telegram topic identifiers are positive Bot API integers.  Core does not
    infer a topic from caller/model text, and it does not support any other
    destination-routing fields for governed delivery.
    """

    if destination_topic_id is None or destination_topic_id == "":
        return None, ""
    if str(platform).lower() != "telegram":
        return None, "governed topic routing is unsupported for this platform"
    try:
        topic_id = int(str(destination_topic_id))
    except (TypeError, ValueError):
        return None, "Telegram topic id was not an integer"
    if topic_id <= 0:
        return None, "Telegram topic id was not positive"
    return str(topic_id), ""


def destination_topic_id_from_metadata(
    *, platform: str, metadata: Any
) -> tuple[Optional[str], str]:
    """Extract one unambiguous logical topic from adapter routing metadata.

    Only the explicit bounded fields Hermes' Telegram adapter uses are
    accepted.  Multiple populated fields must identify the same logical topic;
    arbitrary route dictionaries are not interpreted.
    """

    if not isinstance(metadata, dict):
        return None, ""
    platform_name = str(platform).lower()
    if platform_name != "telegram":
        if any(
            metadata.get(key) not in (None, "")
            for key in (
                "thread_id",
                "message_thread_id",
                "direct_messages_topic_id",
                "telegram_direct_messages_topic_id",
            )
        ):
            return None, "governed topic routing is unsupported for this platform"
        return None, ""

    values = [
        metadata.get(key)
        for key in (
            "thread_id",
            "message_thread_id",
            "direct_messages_topic_id",
            "telegram_direct_messages_topic_id",
        )
        if metadata.get(key) not in (None, "")
    ]
    canonical: set[str] = set()
    for value in values:
        topic_id, error = _canonical_destination_topic_id(platform_name, value)
        if error:
            return None, error
        if topic_id is not None:
            canonical.add(topic_id)
    if len(canonical) > 1:
        return None, "Telegram topic routing fields disagreed"
    return (next(iter(canonical)) if canonical else None), ""


def _prune_locked(now: float) -> None:
    cutoff = now - _RECORD_TTL_SECONDS
    for key in list(_completions):
        rows = _completions[key]
        if not rows or max(row.recorded_at for row in rows.values()) < cutoff:
            _completions.pop(key, None)
            _completion_conflicts.discard(key)
    for key, row in list(_outbound.items()):
        if row.created_at < cutoff:
            _outbound.pop(key, None)
            _reserved_outbound.discard(key)

    # Bound memory even if a caller supplies many unique session/turn IDs.
    while len(_completions) > _MAX_RECORDS:
        oldest = min(
            _completions,
            key=lambda key: max(
                row.recorded_at for row in _completions[key].values()
            ),
        )
        _completions.pop(oldest, None)
        _completion_conflicts.discard(oldest)
    while len(_outbound) > _MAX_RECORDS:
        oldest = min(_outbound, key=lambda key: _outbound[key].created_at)
        _outbound.pop(oldest, None)
        _reserved_outbound.discard(oldest)


def _parse_json_object(value: Any) -> Optional[dict[str, Any]]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_result(
    result: Any, *, force: bool = False
) -> tuple[Optional[dict[str, Any]], Optional[dict[str, Any]]]:
    """Return ``(MCP wrapper, product result)`` from a tool completion.

    Hermes' MCP handler returns ``{"result": ..., "structuredContent": ...}``.
    FastMCP may put the product object in structuredContent, in ``result`` as a
    dict, or in ``result`` as JSON text.  Direct dict/JSON tool handlers remain
    supported for non-MCP tests and plugins.
    """

    if isinstance(result, str):
        # Avoid JSON parsing on the overwhelmingly common ungoverned path.
        if not force and REQUIRED_DELIVERY_CONTRACT not in result:
            return None, None
        parsed = _parse_json_object(result)
    else:
        parsed = deepcopy(result) if isinstance(result, dict) else None
    if parsed is None:
        return None, None

    structured = _parse_json_object(parsed.get("structuredContent"))
    if structured is not None:
        return parsed, structured
    wrapped_result = _parse_json_object(parsed.get("result"))
    if wrapped_result is not None:
        return parsed, wrapped_result
    return parsed, parsed


def record_successful_tool_completion(
    *,
    tool_name: str,
    result: Any,
    status: str,
    session_id: str,
    turn_id: str,
    tool_call_id: str,
    api_request_id: str = "",
    task_id: str = "",
) -> bool:
    """Record a standardized required marker from a direct tool completion.

    Returns ``True`` when the result carried the marker or when a registered
    required tool omitted/malformed it and therefore must fail closed.  Only
    ``ok`` completions are eligible.  Identity in the tool result is
    deliberately not used as the Hermes turn binding; the event's
    session/turn/tool-call IDs are.
    """

    if status != "ok":
        return False
    try:
        from hermes_cli.plugins import get_gateway_delivery_requirements_for_tool

        declared_requirements = get_gateway_delivery_requirements_for_tool(tool_name)
    except Exception:
        declared_requirements = []

    wrapper, product_result = _parse_result(
        result, force=bool(declared_requirements)
    )
    marker = (
        product_result.get("delivery_contract")
        if isinstance(product_result, dict)
        else None
    )
    has_required_marker = (
        isinstance(marker, dict)
        and marker.get("contract") == REQUIRED_DELIVERY_CONTRACT
        and marker.get("required") is True
    )
    if not has_required_marker and not declared_requirements:
        return False

    renderer_id = marker.get("renderer_id") if isinstance(marker, dict) else ""
    renderer_version = (
        marker.get("renderer_version") if isinstance(marker, dict) else ""
    )
    invalid_reasons: list[str] = []
    if not has_required_marker:
        invalid_reasons.append("missing or malformed required delivery marker")
    if len(declared_requirements) > 1:
        invalid_reasons.append("tool matched multiple required renderers")
    elif len(declared_requirements) == 1:
        expected_id, expected_version = declared_requirements[0]
        if renderer_id and renderer_id != expected_id:
            invalid_reasons.append("marker renderer_id did not match registration")
        if renderer_version and renderer_version != expected_version:
            invalid_reasons.append(
                "marker renderer_version did not match registration"
            )
        renderer_id = expected_id
        renderer_version = expected_version
    if not isinstance(renderer_id, str) or not renderer_id.strip():
        invalid_reasons.append("invalid renderer_id")
        renderer_id = ""
    if not isinstance(renderer_version, str) or not renderer_version.strip():
        invalid_reasons.append("invalid renderer_version")
        renderer_version = ""
    if not session_id:
        invalid_reasons.append("missing Hermes session_id")
    if not turn_id:
        invalid_reasons.append("missing Hermes turn_id")
    if not tool_call_id:
        invalid_reasons.append("missing tool_call_id")

    completion_role, final_tool_name = _completion_role(product_result)
    if completion_role or final_tool_name:
        if completion_role not in {"requires_final", "final"}:
            invalid_reasons.append("invalid completion_role")
        if not final_tool_name:
            invalid_reasons.append("missing final_tool_name")
        if completion_role == "requires_final":
            invalid_reasons.append("final required completion not observed")
        if completion_role == "final" and final_tool_name != tool_name:
            invalid_reasons.append("final completion tool did not match final_tool_name")

    # A normal gateway completion always supplies all three bindings.  When a
    # binding is absent, retain the marker under the available session/turn so
    # finalization fails closed rather than treating the result as ungoverned.
    key = (session_id or "<missing-session>", turn_id or "<missing-turn>")
    call_key = tool_call_id or "<missing-tool-call>"
    row = RequiredToolCompletion(
        session_id=session_id,
        turn_id=turn_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        api_request_id=api_request_id,
        task_id=task_id,
        result=deepcopy(product_result or wrapper or {}),
        renderer_id=str(renderer_id).strip(),
        renderer_version=str(renderer_version).strip(),
        recorded_at=time.monotonic(),
        invalid_reason="; ".join(invalid_reasons),
    )
    with _lock:
        _prune_locked(row.recorded_at)
        bucket = _completions.setdefault(key, {})
        if completion_role == "final" and final_tool_name == tool_name:
            # Intermediate governed tools make omission of the final evidence
            # call fail closed.  The exact final completion supersedes only
            # matching obligations from this renderer in the same turn; it
            # never hides unrelated governed completions or conflicts.
            for prior_key, prior_row in list(bucket.items()):
                prior_role, prior_final_tool = _completion_role(prior_row.result)
                if (
                    prior_role == "requires_final"
                    and prior_final_tool == tool_name
                    and prior_row.renderer_id == row.renderer_id
                    and prior_row.renderer_version == row.renderer_version
                ):
                    bucket.pop(prior_key, None)
        prior = bucket.get(call_key)
        if prior is not None and prior != row:
            _completion_conflicts.add(key)
        bucket[call_key] = row
    return True


def record_tool_completion_collection_failure(
    *,
    tool_name: str,
    session_id: str,
    turn_id: str,
    tool_call_id: str,
) -> None:
    """Make a collector exception fail a declared governed tool closed."""

    try:
        from hermes_cli.plugins import get_gateway_delivery_requirements_for_tool

        requirements = get_gateway_delivery_requirements_for_tool(tool_name)
    except Exception:
        requirements = []
    if not requirements:
        return
    renderer_id, renderer_version = requirements[0]
    key = (session_id or "<missing-session>", turn_id or "<missing-turn>")
    row = RequiredToolCompletion(
        session_id=session_id,
        turn_id=turn_id,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        api_request_id="",
        task_id="",
        result={},
        renderer_id=renderer_id,
        renderer_version=renderer_version,
        recorded_at=time.monotonic(),
        invalid_reason="required delivery collector failed",
    )
    with _lock:
        _prune_locked(row.recorded_at)
        _completions.setdefault(key, {})[tool_call_id or "<missing-tool-call>"] = row


def _take_completion(session_id: str, turn_id: str) -> tuple[Optional[RequiredToolCompletion], str]:
    key = (session_id or "<missing-session>", turn_id or "<missing-turn>")
    with _lock:
        _prune_locked(time.monotonic())
        rows = _completions.pop(key, {})
        conflicted = key in _completion_conflicts
        _completion_conflicts.discard(key)
    if conflicted:
        return None, "conflicting required tool completion"
    if not rows:
        return None, ""
    if len(rows) != 1:
        return None, "ambiguous required tool completions"
    return next(iter(rows.values())), ""


def _store_outbound(
    platform: str,
    destination_id: str,
    destination_topic_id: Optional[str],
    payload: str,
    *,
    trusted_disclosure: str,
    optional_prose: str,
) -> PreparedGatewayPayload:
    now = time.monotonic()
    token = secrets.token_urlsafe(24)
    record = _OutboundRecord(
        platform=platform,
        destination_id=str(destination_id),
        destination_topic_id=destination_topic_id,
        payload=payload,
        payload_sha256=_sha256_text(payload),
        trusted_disclosure=trusted_disclosure,
        optional_prose=optional_prose,
        token=token,
        created_at=now,
    )
    with _lock:
        _prune_locked(now)
        _outbound[token] = record
        _reserved_outbound.discard(token)
    return PreparedGatewayPayload(payload, token)


def _fail_closed(
    platform: str,
    destination_id: str,
    destination_topic_id: Optional[str],
) -> PreparedGatewayPayload:
    return _store_outbound(
        platform,
        destination_id,
        destination_topic_id,
        FAIL_CLOSED_TEXT,
        trusted_disclosure=FAIL_CLOSED_TEXT,
        optional_prose="",
    )


def prepare_gateway_delivery(
    *,
    session_id: str,
    turn_id: str,
    response_text: str,
    platform: str,
    destination_id: str,
    destination_topic_id: Any = None,
) -> Optional[PreparedGatewayPayload]:
    """Render and bind the complete payload for one required completion.

    ``None`` means the same turn had no required tool completion.  Any error
    after a required marker was observed returns the fixed fail-closed payload
    and never returns model prose.  ``destination_id`` must be the authenticated
    source destination from the gateway event.  For Telegram,
    ``destination_topic_id`` is the authenticated event's logical topic id.
    Core binds both to the opaque outbound plan but deliberately discloses
    neither to the renderer.
    """

    completion, completion_error = _take_completion(session_id, turn_id)
    if completion is None and not completion_error:
        return None
    canonical_topic_id, route_error = _canonical_destination_topic_id(
        platform, destination_topic_id
    )
    if route_error or not str(destination_id):
        # Return an unregistered token-shaped payload.  Adapter validation then
        # rejects it before any platform call; unsupported routes must not be
        # weakened into a root-chat delivery of even the fixed failure text.
        return PreparedGatewayPayload(FAIL_CLOSED_TEXT, secrets.token_urlsafe(24))
    if completion is None or completion_error or completion.invalid_reason:
        return _fail_closed(platform, destination_id, canonical_topic_id)

    try:
        from hermes_cli.plugins import get_gateway_delivery_renderer

        renderer = get_gateway_delivery_renderer(
            completion.renderer_id,
            completion.renderer_version,
            platform,
        )
        if renderer is None:
            return _fail_closed(platform, destination_id, canonical_topic_id)
        rendered = renderer(
            completion={
                "tool_name": completion.tool_name,
                "result": deepcopy(completion.result),
                "session_id": completion.session_id,
                "turn_id": completion.turn_id,
                "tool_call_id": completion.tool_call_id,
                "api_request_id": completion.api_request_id,
                "task_id": completion.task_id,
            },
            response_text=response_text,
            platform=platform,
        )
    except Exception:
        return _fail_closed(platform, destination_id, canonical_topic_id)

    if not isinstance(rendered, dict):
        return _fail_closed(platform, destination_id, canonical_topic_id)
    if rendered.get("contract") != RENDERED_DELIVERY_CONTRACT:
        return _fail_closed(platform, destination_id, canonical_topic_id)
    if rendered.get("bounded_patterns_checked") is not True:
        return _fail_closed(platform, destination_id, canonical_topic_id)
    trusted = rendered.get("trusted_disclosure")
    optional = rendered.get("optional_prose")
    if not isinstance(trusted, str) or not trusted.strip():
        return _fail_closed(platform, destination_id, canonical_topic_id)
    if optional is not None and not isinstance(optional, str):
        return _fail_closed(platform, destination_id, canonical_topic_id)

    trusted = trusted.strip()
    optional = (optional or "").strip()
    payload = trusted if not optional else f"{trusted}\n\n{optional}"

    # Required payloads are text-only.  Attachment directives would let base
    # gateway processing send content before or outside the validated text.
    forbidden = ("MEDIA:", "[[as_document]]", "![")
    if any(marker in payload for marker in forbidden):
        return _fail_closed(platform, destination_id, canonical_topic_id)

    return _store_outbound(
        platform,
        destination_id,
        canonical_topic_id,
        payload,
        trusted_disclosure=trusted,
        optional_prose=optional,
    )


def apply_required_gateway_delivery(
    agent_result: dict[str, Any],
    *,
    response_text: str,
    platform: str,
    destination_id: str,
    destination_topic_id: Any = None,
) -> Optional[PreparedGatewayPayload]:
    """Prepare a governed payload and make it impossible to skip as streamed.

    Gateway callers use this at every point where a completed agent turn could
    be sent.  A required completion always clears ``already_sent`` because
    untrusted streaming/interim delivery can never satisfy the trusted
    disclosure contract.
    """

    prepared = prepare_gateway_delivery(
        session_id=str(agent_result.get("session_id") or ""),
        turn_id=str(agent_result.get("turn_id") or ""),
        response_text=response_text,
        platform=platform,
        destination_id=destination_id,
        destination_topic_id=destination_topic_id,
    )
    if prepared is None:
        return None
    agent_result["already_sent"] = False
    agent_result["required_gateway_delivery"] = True
    agent_result["required_gateway_delivery_token"] = (
        prepared.required_delivery_token
    )
    agent_result["final_response"] = prepared
    return prepared


def should_suppress_model_interims(platform: str) -> bool:
    """Return whether registered policy requires whole-turn prose buffering."""

    try:
        from hermes_cli.plugins import has_gateway_delivery_renderer_for_platform

        return has_gateway_delivery_renderer_for_platform(platform)
    except Exception:
        # Registry failures must not reopen model streaming on a platform that
        # may have a required renderer.  Buffering is the fail-closed default;
        # the ordinary final send remains available for ungoverned turns.
        return True


def validate_outbound_payload(
    *,
    platform: str,
    destination_id: str,
    destination_topic_id: Any = None,
    content: str,
    token: str,
) -> OutboundValidation:
    """Validate the complete payload before a platform performs any send."""

    with _lock:
        _prune_locked(time.monotonic())
        row = _outbound.get(token) if token else None
        reserved = token in _reserved_outbound
    if row is None:
        # A token-shaped request is governed even when the plan is stale or
        # already consumed; never degrade it into an ordinary send.
        return OutboundValidation(
            governed=bool(token),
            valid=not bool(token),
            token=token,
            error="required delivery plan is missing or expired" if token else "",
        )
    if reserved:
        return OutboundValidation(
            governed=True,
            valid=False,
            token=token,
            error="required delivery plan is already reserved",
        )
    canonical_topic_id, route_error = _canonical_destination_topic_id(
        platform, destination_topic_id
    )
    if (
        route_error
        or row.platform != platform
        or row.destination_id != str(destination_id)
        or row.destination_topic_id != canonical_topic_id
    ):
        return OutboundValidation(
            governed=True,
            valid=False,
            token=token,
            error="required delivery destination did not match the prepared plan",
        )
    if not isinstance(content, str):
        return OutboundValidation(
            governed=True, valid=False, error="required payload must be text"
        )
    if _sha256_text(content) != row.payload_sha256 or content != row.payload:
        return OutboundValidation(
            governed=True,
            valid=False,
            token=row.token,
            error="complete required payload did not match the prepared digest",
        )
    return OutboundValidation(
        governed=True,
        valid=True,
        token=row.token,
        trusted_disclosure=row.trusted_disclosure,
        optional_prose=row.optional_prose,
    )


def reserve_outbound_payload(
    *,
    platform: str,
    destination_id: str,
    destination_topic_id: Any = None,
    content: str,
    token: str,
) -> OutboundValidation:
    """Atomically validate and reserve a plan before the first platform call.

    Reservation closes concurrent and direct-adapter replay.  The platform
    adapter must complete or invalidate the exact reserved plan before
    returning; any ambiguous outcome is invalidated rather than made reusable.
    """

    with _lock:
        _prune_locked(time.monotonic())
        row = _outbound.get(token) if token else None
        if row is None:
            return OutboundValidation(
                governed=bool(token),
                valid=not bool(token),
                token=token,
                error=(
                    "required delivery plan is missing or expired"
                    if token else ""
                ),
            )
        if token in _reserved_outbound:
            return OutboundValidation(
                governed=True,
                valid=False,
                token=token,
                error="required delivery plan is already reserved",
            )
        canonical_topic_id, route_error = _canonical_destination_topic_id(
            platform, destination_topic_id
        )
        if (
            route_error
            or row.platform != platform
            or row.destination_id != str(destination_id)
            or row.destination_topic_id != canonical_topic_id
        ):
            return OutboundValidation(
                governed=True,
                valid=False,
                token=token,
                error=(
                    "required delivery destination did not match the "
                    "prepared plan"
                ),
            )
        if not isinstance(content, str):
            return OutboundValidation(
                governed=True,
                valid=False,
                token=token,
                error="required payload must be text",
            )
        if _sha256_text(content) != row.payload_sha256 or content != row.payload:
            return OutboundValidation(
                governed=True,
                valid=False,
                token=row.token,
                error=(
                    "complete required payload did not match the prepared "
                    "digest"
                ),
            )
        _reserved_outbound.add(token)
        return OutboundValidation(
            governed=True,
            valid=True,
            token=row.token,
            trusted_disclosure=row.trusted_disclosure,
            optional_prose=row.optional_prose,
        )


def complete_outbound_payload(
    *,
    platform: str,
    destination_id: str,
    destination_topic_id: Any = None,
    content: str,
    token: str,
) -> bool:
    """Consume a successfully sent payload when destination and digest match."""

    with _lock:
        _prune_locked(time.monotonic())
        row = _outbound.get(token) if token else None
        if row is None:
            return False
        if not isinstance(content, str):
            return False
        if _sha256_text(content) != row.payload_sha256 or content != row.payload:
            return False
        canonical_topic_id, route_error = _canonical_destination_topic_id(
            platform, destination_topic_id
        )
        if (
            route_error
            or row.platform != platform
            or row.destination_id != str(destination_id)
            or row.destination_topic_id != canonical_topic_id
        ):
            return False
        _outbound.pop(token, None)
        _reserved_outbound.discard(token)
        return True


def invalidate_outbound_payload(
    *,
    platform: str,
    destination_id: str,
    destination_topic_id: Any = None,
    content: str,
    token: str,
) -> bool:
    """Invalidate an exact plan after any attempted governed delivery.

    An ambiguous first call or partial disclosure cannot be retried safely
    because Telegram may already have accepted content. Removing the plan
    makes every fallback, retry, and replay with the old token fail before
    another Bot API call.
    """

    return complete_outbound_payload(
        platform=platform,
        destination_id=destination_id,
        destination_topic_id=destination_topic_id,
        content=content,
        token=token,
    )


def _reset_required_delivery_state_for_tests() -> None:
    with _lock:
        _completions.clear()
        _completion_conflicts.clear()
        _outbound.clear()
        _reserved_outbound.clear()
