"""Behavior contracts for fail-closed required gateway delivery."""

from __future__ import annotations

import json

import pytest

from gateway.required_delivery import (
    FAIL_CLOSED_TEXT,
    REQUIRED_DELIVERY_CONTRACT,
    RENDERED_DELIVERY_CONTRACT,
    _reset_required_delivery_state_for_tests,
    apply_required_gateway_delivery,
    complete_outbound_payload,
    prepare_gateway_delivery,
    record_successful_tool_completion,
    should_suppress_model_interims,
    validate_outbound_payload,
)
from hermes_cli import plugins as plugin_module
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest


RENDERER_ID = "openhealthatlas-recovery-delivery"
RENDERER_VERSION = "1.0.0"
SNAPSHOT_TOOL = "openhealthatlas_recovery_snapshot"
DETAIL_TOOL = "openhealthatlas_recovery_detail"
SNAPSHOT_OBSERVED = f"mcp_openhealthatlas_fictional_{SNAPSHOT_TOOL}"
DETAIL_OBSERVED = f"mcp_openhealthatlas_fictional_{DETAIL_TOOL}"
GENERIC_RENDERER_ID = "openhealthatlas-generic-evidence-delivery"
GENERIC_CATALOG = "mcp_openhealthatlas_fictional_openhealthatlas_health_catalog"
GENERIC_QUERY = "mcp_openhealthatlas_fictional_openhealthatlas_health_query"
GENERIC_ANALYZE = "mcp_openhealthatlas_fictional_openhealthatlas_health_analyze"
GENERIC_EVIDENCE = "mcp_openhealthatlas_fictional_openhealthatlas_health_evidence"


@pytest.fixture(autouse=True)
def _isolated_required_delivery(monkeypatch):
    manager = PluginManager()
    monkeypatch.setattr(plugin_module, "_plugin_manager", manager)
    _reset_required_delivery_state_for_tests()
    yield manager
    _reset_required_delivery_state_for_tests()


def _register(manager: PluginManager, callback):
    context = PluginContext(
        PluginManifest(
            name=RENDERER_ID,
            key=RENDERER_ID,
            version=RENDERER_VERSION,
        ),
        manager,
    )
    context.register_gateway_delivery_renderer(
        RENDERER_ID,
        RENDERER_VERSION,
        callback,
        platforms=("telegram",),
        required_tool_names=(SNAPSHOT_OBSERVED, DETAIL_OBSERVED),
    )


def _product_result(*, marker=True):
    payload = {
        "ok": True,
        "data_class": "fictional",
        "openhealthatlas_turn_id": "product-domain-turn-not-hermes-turn",
        "required_disclosure": {"status": "insufficient_data"},
    }
    if marker:
        payload["delivery_contract"] = {
            "contract": REQUIRED_DELIVERY_CONTRACT,
            "required": True,
            "renderer_id": RENDERER_ID,
            "renderer_version": RENDERER_VERSION,
        }
    return payload


def _mcp_completion_wrapper(payload):
    # This is the production tools/mcp_tool.py success shape when FastMCP
    # supplies model-oriented text plus structuredContent.
    return json.dumps(
        {
            "result": json.dumps(payload, sort_keys=True),
            "structuredContent": payload,
        },
        sort_keys=True,
    )


def _record(payload, *, session="session-1", turn="hermes-turn-1", call="call-1"):
    return record_successful_tool_completion(
        tool_name=SNAPSHOT_OBSERVED,
        result=_mcp_completion_wrapper(payload),
        status="ok",
        session_id=session,
        turn_id=turn,
        tool_call_id=call,
        api_request_id=f"{turn}:api:1",
        task_id="task-1",
    )


def _register_generic(manager: PluginManager, callback):
    context = PluginContext(
        PluginManifest(
            name=GENERIC_RENDERER_ID,
            key=GENERIC_RENDERER_ID,
            version="1.0.0",
        ),
        manager,
    )
    context.register_gateway_delivery_renderer(
        GENERIC_RENDERER_ID,
        "1.0.0",
        callback,
        platforms=("telegram",),
        required_tool_names=(
            GENERIC_CATALOG, GENERIC_QUERY, GENERIC_ANALYZE, GENERIC_EVIDENCE,
        ),
    )


def _generic_product_result(tool_name: str, role: str) -> dict:
    return {
        "contract": "openhealthatlas-hermes-surface-v1",
        "operation": tool_name.removeprefix("mcp_openhealthatlas_fictional_openhealthatlas_"),
        "delivery_contract": {
            "contract": REQUIRED_DELIVERY_CONTRACT,
            "required": True,
            "renderer_id": GENERIC_RENDERER_ID,
            "renderer_version": "1.0.0",
            "completion_role": role,
            "final_tool_name": GENERIC_EVIDENCE,
        },
    }


def _record_generic(
    tool_name: str,
    role: str,
    *,
    call: str,
    session: str = "generic-session",
    turn: str = "generic-turn",
) -> bool:
    return record_successful_tool_completion(
        tool_name=tool_name,
        result=_mcp_completion_wrapper(_generic_product_result(tool_name, role)),
        status="ok",
        session_id=session,
        turn_id=turn,
        tool_call_id=call,
    )


def test_generic_intermediate_tool_requires_final_evidence_or_fails_closed(
    _isolated_required_delivery,
):
    _register_generic(
        _isolated_required_delivery,
        lambda **_: pytest.fail("intermediate completion must not render"),
    )
    assert _record_generic(GENERIC_QUERY, "requires_final", call="query-1")

    prepared = prepare_gateway_delivery(
        session_id="generic-session",
        turn_id="generic-turn",
        response_text="ungoverned model prose",
        platform="telegram",
        destination_id="chat-generic",
    )

    assert str(prepared) == FAIL_CLOSED_TEXT
    assert "ungoverned model prose" not in str(prepared)


def test_generic_final_evidence_supersedes_same_renderer_obligations(
    _isolated_required_delivery,
):
    observed = {}

    def renderer(**kwargs):
        observed.update(kwargs)
        return {
            "contract": RENDERED_DELIVERY_CONTRACT,
            "trusted_disclosure": "GENERIC DISCLOSURE",
            "optional_prose": "Hermes interpretation\nBounded explanation.",
            "bounded_patterns_checked": True,
        }

    _register_generic(_isolated_required_delivery, renderer)
    assert _record_generic(GENERIC_CATALOG, "requires_final", call="catalog-1")
    assert _record_generic(GENERIC_QUERY, "requires_final", call="query-1")
    assert _record_generic(GENERIC_ANALYZE, "requires_final", call="analyze-1")
    assert _record_generic(GENERIC_EVIDENCE, "final", call="evidence-1")

    prepared = prepare_gateway_delivery(
        session_id="generic-session",
        turn_id="generic-turn",
        response_text="Bounded explanation.",
        platform="telegram",
        destination_id="chat-generic",
    )

    assert str(prepared).startswith("GENERIC DISCLOSURE\n\n")
    assert observed["completion"]["tool_name"] == GENERIC_EVIDENCE
    assert observed["completion"]["tool_call_id"] == "evidence-1"


def test_actual_mcp_completion_is_bound_to_hermes_turn_not_product_turn(
    _isolated_required_delivery,
):
    observed = {}

    def renderer(**kwargs):
        observed.update(kwargs)
        return {
            "contract": RENDERED_DELIVERY_CONTRACT,
            "trusted_disclosure": "TRUSTED DISCLOSURE",
            "optional_prose": "Optional Hermes explanation.",
            "bounded_patterns_checked": True,
        }

    _register(_isolated_required_delivery, renderer)
    assert _record(_product_result()) is True

    prepared = prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="Optional Hermes explanation.",
        platform="telegram",
        destination_id="chat-1",
    )

    assert str(prepared) == "TRUSTED DISCLOSURE\n\nOptional Hermes explanation."
    assert observed["completion"]["turn_id"] == "hermes-turn-1"
    assert (
        observed["completion"]["result"]["openhealthatlas_turn_id"]
        == "product-domain-turn-not-hermes-turn"
    )
    assert observed["completion"]["tool_call_id"] == "call-1"
    assert observed["completion"]["tool_name"] == SNAPSHOT_OBSERVED
    assert observed["response_text"] == "Optional Hermes explanation."
    assert observed["platform"] == "telegram"
    assert "destination_id" not in observed

    validation = validate_outbound_payload(
        platform="telegram",
        destination_id="chat-1",
        content=str(prepared),
        token=prepared.required_delivery_token,
    )
    assert validation.valid is True
    assert validation.trusted_disclosure == "TRUSTED DISCLOSURE"
    assert validation.optional_prose == "Optional Hermes explanation."


def test_tool_call_bridge_preserves_required_delivery_turn_binding(
    _isolated_required_delivery,
    monkeypatch,
):
    """The real deferred-MCP bridge must not orphan the governed completion."""

    import model_tools
    from tools import tool_search as tool_search_module

    observed = {}

    def renderer(**kwargs):
        observed.update(kwargs)
        return {
            "contract": RENDERED_DELIVERY_CONTRACT,
            "trusted_disclosure": "COMPACT TRUSTED DISCLOSURE",
            "optional_prose": "Hermes interpretation\n\nBounded explanation.",
            "bounded_patterns_checked": True,
        }

    _register(_isolated_required_delivery, renderer)
    monkeypatch.setattr(
        tool_search_module,
        "is_deferrable_tool_name",
        lambda name: name == SNAPSHOT_OBSERVED,
    )
    monkeypatch.setattr(
        model_tools,
        "get_tool_definitions",
        lambda **_: [{
            "type": "function",
            "function": {
                "name": SNAPSHOT_OBSERVED,
                "description": "Bounded fictional Recovery snapshot",
                "parameters": {"type": "object", "properties": {}},
            },
        }],
    )
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda name, args, **_: _mcp_completion_wrapper(_product_result()),
    )

    result = model_tools.handle_function_call(
        function_name="tool_call",
        function_args={"name": SNAPSHOT_OBSERVED, "arguments": {}},
        task_id="task-bridge",
        session_id="session-bridge",
        turn_id="turn-bridge",
        tool_call_id="call-bridge",
        api_request_id="request-bridge",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
    )
    assert json.loads(result)["structuredContent"]["delivery_contract"][
        "required"
    ] is True

    prepared = prepare_gateway_delivery(
        session_id="session-bridge",
        turn_id="turn-bridge",
        response_text="Bounded explanation.",
        platform="telegram",
        destination_id="chat-bridge",
    )

    assert str(prepared).startswith("COMPACT TRUSTED DISCLOSURE\n\n")
    assert "Hermes interpretation" in str(prepared)
    assert observed["completion"]["session_id"] == "session-bridge"
    assert observed["completion"]["turn_id"] == "turn-bridge"
    assert observed["completion"]["tool_call_id"] == "call-bridge"
    assert observed["completion"]["api_request_id"] == "request-bridge"
    assert observed["completion"]["tool_name"] == SNAPSHOT_OBSERVED


def test_registered_governed_tool_without_marker_fails_closed(
    _isolated_required_delivery,
):
    _register(
        _isolated_required_delivery,
        lambda **_: pytest.fail("malformed result must not reach renderer"),
    )
    assert _record(_product_result(marker=False)) is True

    prepared = prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="model prose must disappear",
        platform="telegram",
        destination_id="chat-1",
    )

    assert str(prepared) == FAIL_CLOSED_TEXT
    assert "model prose" not in str(prepared)


def test_required_tool_registration_does_not_suffix_match_another_mcp_server(
    _isolated_required_delivery,
):
    _register(_isolated_required_delivery, lambda **_: {})

    recorded = record_successful_tool_completion(
        tool_name=f"mcp_other_{SNAPSHOT_TOOL}",
        result=_mcp_completion_wrapper(_product_result(marker=False)),
        status="ok",
        session_id="session-1",
        turn_id="hermes-turn-1",
        tool_call_id="call-1",
    )

    assert recorded is False
    assert prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="ordinary response",
        platform="telegram",
        destination_id="chat-1",
    ) is None


def test_required_delivery_overrides_already_sent_early_return(
    _isolated_required_delivery,
):
    _register(
        _isolated_required_delivery,
        lambda **_: {
            "contract": RENDERED_DELIVERY_CONTRACT,
            "trusted_disclosure": "DISCLOSURE",
            "optional_prose": "PROSE",
            "bounded_patterns_checked": True,
        },
    )
    _record(_product_result())
    agent_result = {
        "session_id": "session-1",
        "turn_id": "hermes-turn-1",
        "already_sent": True,
        "final_response": "PROSE",
    }

    prepared = apply_required_gateway_delivery(
        agent_result,
        response_text="PROSE",
        platform="telegram",
        destination_id="chat-1",
    )

    assert str(prepared) == "DISCLOSURE\n\nPROSE"
    assert agent_result["already_sent"] is False
    assert agent_result["required_gateway_delivery"] is True
    assert agent_result["final_response"] is prepared


def test_model_media_directive_cannot_escape_before_trusted_disclosure(
    _isolated_required_delivery,
):
    _register(
        _isolated_required_delivery,
        lambda **_: {
            "contract": RENDERED_DELIVERY_CONTRACT,
            "trusted_disclosure": "DISCLOSURE",
            "optional_prose": "MEDIA:/private/untrusted.mp3",
            "bounded_patterns_checked": True,
        },
    )
    _record(_product_result())

    prepared = prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="MEDIA:/private/untrusted.mp3",
        platform="telegram",
        destination_id="chat-1",
    )

    assert str(prepared) == FAIL_CLOSED_TEXT
    assert "MEDIA:" not in str(prepared)


def test_marker_without_installed_renderer_fails_closed():
    assert _record(_product_result()) is True
    prepared = prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="untrusted prose",
        platform="telegram",
        destination_id="chat-1",
    )
    assert str(prepared) == FAIL_CLOSED_TEXT


def test_renderer_exception_and_invalid_return_fail_closed(
    _isolated_required_delivery,
):
    def renderer(**_):
        raise RuntimeError("renderer failed")

    _register(_isolated_required_delivery, renderer)
    _record(_product_result())
    prepared = prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="untrusted prose",
        platform="telegram",
        destination_id="chat-1",
    )
    assert str(prepared) == FAIL_CLOSED_TEXT


def test_collector_exception_for_declared_tool_is_recorded_fail_closed(
    _isolated_required_delivery,
    monkeypatch,
):
    import gateway.required_delivery as required_delivery
    from model_tools import _emit_post_tool_call_hook

    _register(
        _isolated_required_delivery,
        lambda **_: pytest.fail("collector failure must not reach renderer"),
    )

    def fail_collector(**_):
        raise RuntimeError("synthetic collector failure")

    monkeypatch.setattr(
        required_delivery,
        "record_successful_tool_completion",
        fail_collector,
    )
    _emit_post_tool_call_hook(
        function_name=SNAPSHOT_OBSERVED,
        function_args={},
        result=_mcp_completion_wrapper(_product_result()),
        status="ok",
        session_id="session-1",
        turn_id="hermes-turn-1",
        tool_call_id="call-1",
    )

    prepared = prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="model prose must disappear",
        platform="telegram",
        destination_id="chat-1",
    )
    assert str(prepared) == FAIL_CLOSED_TEXT
    assert "model prose" not in str(prepared)


def test_concurrent_same_destination_plans_are_token_isolated(
    _isolated_required_delivery,
):
    def renderer(**kwargs):
        turn = kwargs["completion"]["turn_id"]
        return {
            "contract": RENDERED_DELIVERY_CONTRACT,
            "trusted_disclosure": f"DISCLOSURE {turn}",
            "optional_prose": "",
            "bounded_patterns_checked": True,
        }

    _register(_isolated_required_delivery, renderer)
    _record(_product_result(), session="s1", turn="t1", call="c1")
    _record(_product_result(), session="s2", turn="t2", call="c2")
    first = prepare_gateway_delivery(
        session_id="s1",
        turn_id="t1",
        response_text="p1",
        platform="telegram",
        destination_id="same-chat",
    )
    second = prepare_gateway_delivery(
        session_id="s2",
        turn_id="t2",
        response_text="p2",
        platform="telegram",
        destination_id="same-chat",
    )
    assert first.required_delivery_token != second.required_delivery_token
    assert validate_outbound_payload(
        platform="telegram",
        destination_id="same-chat",
        content=str(first),
        token=first.required_delivery_token,
    ).valid
    assert validate_outbound_payload(
        platform="telegram",
        destination_id="same-chat",
        content=str(second),
        token=second.required_delivery_token,
    ).valid


def test_payload_digest_destination_and_double_send_are_fail_closed(
    _isolated_required_delivery,
):
    _register(
        _isolated_required_delivery,
        lambda **_: {
            "contract": RENDERED_DELIVERY_CONTRACT,
            "trusted_disclosure": "DISCLOSURE",
            "optional_prose": "PROSE",
            "bounded_patterns_checked": True,
        },
    )
    _record(_product_result())
    prepared = prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="PROSE",
        platform="telegram",
        destination_id="chat-1",
    )
    token = prepared.required_delivery_token
    assert not validate_outbound_payload(
        platform="telegram",
        destination_id="chat-1",
        content=str(prepared) + " altered",
        token=token,
    ).valid
    assert not validate_outbound_payload(
        platform="telegram",
        destination_id="chat-2",
        content=str(prepared),
        token=token,
    ).valid
    assert complete_outbound_payload(
        platform="telegram",
        destination_id="chat-1",
        content=str(prepared),
        token=token,
    )
    replay = validate_outbound_payload(
        platform="telegram",
        destination_id="chat-1",
        content=str(prepared),
        token=token,
    )
    assert replay.governed is True
    assert replay.valid is False


def test_telegram_topic_is_part_of_destination_binding(
    _isolated_required_delivery,
):
    _register(
        _isolated_required_delivery,
        lambda **_: {
            "contract": RENDERED_DELIVERY_CONTRACT,
            "trusted_disclosure": "DISCLOSURE",
            "optional_prose": "",
            "bounded_patterns_checked": True,
        },
    )
    _record(_product_result())
    prepared = prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="",
        platform="telegram",
        destination_id="chat-1",
        destination_topic_id="77",
    )

    assert not validate_outbound_payload(
        platform="telegram",
        destination_id="chat-1",
        destination_topic_id="78",
        content=str(prepared),
        token=prepared.required_delivery_token,
    ).valid
    assert validate_outbound_payload(
        platform="telegram",
        destination_id="chat-1",
        destination_topic_id=77,
        content=str(prepared),
        token=prepared.required_delivery_token,
    ).valid


def test_unsupported_telegram_topic_route_is_fail_closed_before_send(
    _isolated_required_delivery,
):
    _register(_isolated_required_delivery, lambda **_: {})
    _record(_product_result())
    prepared = prepare_gateway_delivery(
        session_id="session-1",
        turn_id="hermes-turn-1",
        response_text="model prose",
        platform="telegram",
        destination_id="chat-1",
        destination_topic_id="not-a-topic",
    )

    assert str(prepared) == FAIL_CLOSED_TEXT
    assert not validate_outbound_payload(
        platform="telegram",
        destination_id="chat-1",
        content=str(prepared),
        token=prepared.required_delivery_token,
    ).valid


def test_registered_telegram_renderer_suppresses_whole_turn_model_interims(
    _isolated_required_delivery,
    monkeypatch,
):
    _register(_isolated_required_delivery, lambda **_: {})
    assert should_suppress_model_interims("telegram") is True
    assert should_suppress_model_interims("discord") is False

    monkeypatch.setattr(
        plugin_module,
        "has_gateway_delivery_renderer_for_platform",
        lambda _platform: (_ for _ in ()).throw(RuntimeError("registry failed")),
    )
    assert should_suppress_model_interims("telegram") is True
