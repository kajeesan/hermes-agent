"""Telegram end-to-end send-boundary tests for required delivery plans."""

from __future__ import annotations

import json
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _ensure_telegram_mock():
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    mod = MagicMock()
    mod.ext.ContextTypes.DEFAULT_TYPE = type(None)
    mod.constants.ParseMode.MARKDOWN = "Markdown"
    mod.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    mod.constants.ParseMode.HTML = "HTML"
    mod.constants.ChatType.PRIVATE = "private"
    mod.constants.ChatType.GROUP = "group"
    mod.constants.ChatType.SUPERGROUP = "supergroup"
    mod.constants.ChatType.CHANNEL = "channel"
    mod.error.NetworkError = type("NetworkError", (OSError,), {})
    mod.error.TimedOut = type("TimedOut", (OSError,), {})
    mod.error.BadRequest = type("BadRequest", (mod.error.NetworkError,), {})
    for name in ("telegram", "telegram.ext", "telegram.constants", "telegram.request"):
        sys.modules.setdefault(name, mod)
    sys.modules.setdefault("telegram.error", mod.error)


_ensure_telegram_mock()

from gateway.config import PlatformConfig
from gateway.required_delivery import (
    REQUIRED_DELIVERY_CONTRACT,
    RENDERED_DELIVERY_CONTRACT,
    _reset_required_delivery_state_for_tests,
    prepare_gateway_delivery,
    record_successful_tool_completion,
)
from hermes_cli import plugins as plugin_module
from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
from plugins.platforms.telegram.adapter import TelegramAdapter
from telegram.error import BadRequest, NetworkError, TimedOut


RENDERER_ID = "openhealthatlas-recovery-delivery"
VERSION = "1.0.0"
TOOL = "openhealthatlas_recovery_snapshot"
OBSERVED_TOOL = f"mcp_openhealthatlas_fictional_{TOOL}"


@pytest.fixture(autouse=True)
def _state(monkeypatch):
    manager = PluginManager()
    monkeypatch.setattr(plugin_module, "_plugin_manager", manager)
    _reset_required_delivery_state_for_tests()
    yield manager
    _reset_required_delivery_state_for_tests()


def _prepared(
    manager,
    *,
    trusted: str,
    prose: str,
    chat_id: str = "123",
    topic_id: str | None = None,
):
    context = PluginContext(
        PluginManifest(name=RENDERER_ID, key=RENDERER_ID, version=VERSION),
        manager,
    )
    context.register_gateway_delivery_renderer(
        RENDERER_ID,
        VERSION,
        lambda **_: {
            "contract": RENDERED_DELIVERY_CONTRACT,
            "trusted_disclosure": trusted,
            "optional_prose": prose,
            "bounded_patterns_checked": True,
        },
        platforms=("telegram",),
        required_tool_names=(OBSERVED_TOOL,),
    )
    product = {
        "ok": True,
        "delivery_contract": {
            "contract": REQUIRED_DELIVERY_CONTRACT,
            "required": True,
            "renderer_id": RENDERER_ID,
            "renderer_version": VERSION,
        },
        "required_disclosure": {"status": "insufficient_data"},
    }
    mcp_result = json.dumps(
        {"result": json.dumps(product), "structuredContent": product}
    )
    assert record_successful_tool_completion(
        tool_name=OBSERVED_TOOL,
        result=mcp_result,
        status="ok",
        session_id="session",
        turn_id="turn",
        tool_call_id="call",
    )
    return prepare_gateway_delivery(
        session_id="session",
        turn_id="turn",
        response_text=prose,
        platform="telegram",
        destination_id=chat_id,
        destination_topic_id=topic_id,
    )


def _adapter():
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _message(message_id: int):
    result = MagicMock()
    result.message_id = message_id
    return result


@pytest.mark.asyncio
async def test_complete_plan_validated_before_send_and_disclosure_chunks_precede_prose(
    _state,
):
    prepared = _prepared(
        _state,
        trusted="T" * 46,
        prose="P" * 36,
    )
    adapter = _adapter()
    adapter.MAX_MESSAGE_LENGTH = 20
    sent = []

    async def send_message(**kwargs):
        sent.append(kwargs["text"])
        return _message(len(sent))

    adapter._bot.send_message = AsyncMock(side_effect=send_message)
    metadata = {
        "required_delivery_token": prepared.required_delivery_token,
        "notify": True,
    }
    result = await adapter._send_with_retry(
        chat_id="123", content=str(prepared), metadata=metadata, base_delay=0
    )

    assert result.success is True
    first_prose_index = next(i for i, text in enumerate(sent) if "P" in text)
    assert first_prose_index >= 2
    assert all("P" not in text for text in sent[:first_prose_index])
    assert all("T" not in text for text in sent[first_prose_index:])


@pytest.mark.asyncio
async def test_mismatched_complete_payload_sends_nothing(_state):
    prepared = _prepared(_state, trusted="DISCLOSURE", prose="PROSE")
    adapter = _adapter()
    adapter._bot.send_message = AsyncMock(return_value=_message(1))

    result = await adapter.send(
        chat_id="123",
        content=str(prepared) + " altered",
        metadata={"required_delivery_token": prepared.required_delivery_token},
    )

    assert result.success is False
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_retry_before_first_send_then_success_consumes_plan(_state):
    prepared = _prepared(_state, trusted="DISCLOSURE", prose="")
    adapter = _adapter()
    adapter._bot.send_message = AsyncMock(
        side_effect=[NetworkError("connection dropped"), _message(1)]
    )
    metadata = {
        "required_delivery_token": prepared.required_delivery_token,
        "notify": True,
    }
    with patch("plugins.platforms.telegram.adapter.asyncio.sleep", new=AsyncMock()):
        result = await adapter._send_with_retry(
            chat_id="123", content=str(prepared), metadata=metadata, base_delay=0
        )
    assert result.success is True
    assert adapter._bot.send_message.await_count == 2

    before = adapter._bot.send_message.await_count
    replay = await adapter.send(
        chat_id="123", content=str(prepared), metadata=metadata
    )
    assert replay.success is False
    assert adapter._bot.send_message.await_count == before


@pytest.mark.asyncio
async def test_direct_send_success_consumes_plan_before_return(_state):
    prepared = _prepared(_state, trusted="DISCLOSURE", prose="")
    adapter = _adapter()
    adapter._bot.send_message = AsyncMock(return_value=_message(1))
    metadata = {
        "required_delivery_token": prepared.required_delivery_token,
        "notify": True,
    }

    delivered = await adapter.send(
        chat_id="123", content=str(prepared), metadata=metadata
    )
    assert delivered.success is True
    assert adapter._bot.send_message.await_count == 1

    replay = await adapter.send(
        chat_id="123", content=str(prepared), metadata=metadata
    )
    assert replay.success is False
    assert adapter._bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_ambiguous_first_chunk_timeout_invalidates_plan(_state):
    prepared = _prepared(_state, trusted="DISCLOSURE", prose="")
    adapter = _adapter()
    adapter._bot.send_message = AsyncMock(
        side_effect=TimedOut("ambiguous Telegram timeout")
    )
    metadata = {
        "required_delivery_token": prepared.required_delivery_token,
        "notify": True,
    }

    attempted = await adapter.send(
        chat_id="123", content=str(prepared), metadata=metadata
    )
    assert attempted.success is False
    assert adapter._bot.send_message.await_count == 1

    replay = await adapter.send(
        chat_id="123", content=str(prepared), metadata=metadata
    )
    assert replay.success is False
    assert adapter._bot.send_message.await_count == 1


@pytest.mark.asyncio
async def test_prepared_token_cannot_be_dropped_into_unguarded_send(_state):
    prepared = _prepared(_state, trusted="DISCLOSURE", prose="PROSE")
    adapter = _adapter()
    adapter._bot.send_message = AsyncMock(return_value=_message(1))

    # The standard retry seam recovers the opaque token from the prepared
    # payload type even when its caller supplies no token metadata.
    result = await adapter._send_with_retry(
        chat_id="123",
        content=prepared,
        metadata={"notify": True},
        base_delay=0,
    )
    assert result.success is True
    assert adapter._bot.send_message.await_count == 2

    # Completion consumed the plan; replay with the old prepared object fails
    # before another Bot API call.
    before = adapter._bot.send_message.await_count
    replay = await adapter.send(chat_id="123", content=prepared, metadata={})
    assert replay.success is False
    assert adapter._bot.send_message.await_count == before


@pytest.mark.asyncio
async def test_partial_governed_send_is_not_retried_or_plaintext_resent(_state):
    prepared = _prepared(_state, trusted="DISCLOSURE", prose="PROSE")
    adapter = _adapter()
    adapter._bot.send_message = AsyncMock(
        side_effect=[_message(1), BadRequest("permanent failure")]
    )
    metadata = {
        "required_delivery_token": prepared.required_delivery_token,
        "notify": True,
    }

    result = await adapter._send_with_retry(
        chat_id="123", content=str(prepared), metadata=metadata, base_delay=0
    )

    assert result.success is False
    # The disclosure was sent once; validation blocked Base's plain-text
    # fallback before it could reach the Bot API.
    assert adapter._bot.send_message.await_count == 2
    first_text = adapter._bot.send_message.await_args_list[0].kwargs["text"]
    assert "DISCLOSURE" in first_text

    # The partial send invalidated the plan, so an explicit replay cannot send
    # either the disclosure or the optional prose again.
    replay = await adapter.send(
        chat_id="123", content=str(prepared), metadata=metadata
    )
    assert replay.success is False
    assert adapter._bot.send_message.await_count == 2


@pytest.mark.asyncio
async def test_topic_redirect_and_conflicting_topic_fields_send_nothing(_state):
    prepared = _prepared(
        _state, trusted="DISCLOSURE", prose="", topic_id="77"
    )
    adapter = _adapter()
    adapter._bot.send_message = AsyncMock(return_value=_message(1))

    redirected = await adapter.send(
        chat_id="123",
        content=str(prepared),
        metadata={
            "required_delivery_token": prepared.required_delivery_token,
            "thread_id": "78",
        },
    )
    assert redirected.success is False
    adapter._bot.send_message.assert_not_awaited()

    conflicting = await adapter.send(
        chat_id="123",
        content=str(prepared),
        metadata={
            "required_delivery_token": prepared.required_delivery_token,
            "thread_id": "77",
            "direct_messages_topic_id": "78",
        },
    )
    assert conflicting.success is False
    adapter._bot.send_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_governed_topic_failure_never_falls_back_to_root_chat(_state):
    prepared = _prepared(
        _state,
        trusted="DISCLOSURE",
        prose="",
        chat_id="-100123",
        topic_id="77",
    )
    adapter = _adapter()
    adapter._bot.send_message = AsyncMock(
        side_effect=BadRequest("Message thread not found")
    )

    result = await adapter._send_with_retry(
        chat_id="-100123",
        content=str(prepared),
        metadata={
            "required_delivery_token": prepared.required_delivery_token,
            "thread_id": "77",
            "notify": True,
        },
        base_delay=0,
    )

    assert result.success is False
    assert adapter._bot.send_message.await_count >= 1
    assert all(
        call.kwargs["message_thread_id"] == 77
        for call in adapter._bot.send_message.await_args_list
    )
