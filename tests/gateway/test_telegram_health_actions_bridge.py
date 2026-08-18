"""Bounded tests for the reconstructed live Telegram health-action bridge."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_telegram_mock() -> None:
    if "telegram" in sys.modules and hasattr(sys.modules["telegram"], "__file__"):
        return
    module = MagicMock()
    module.ext.ContextTypes.DEFAULT_TYPE = type(None)
    module.constants.ParseMode.MARKDOWN = "Markdown"
    module.constants.ParseMode.MARKDOWN_V2 = "MarkdownV2"
    module.constants.ParseMode.HTML = "HTML"
    module.constants.ChatType.PRIVATE = "private"
    module.constants.ChatType.GROUP = "group"
    module.constants.ChatType.SUPERGROUP = "supergroup"
    module.constants.ChatType.CHANNEL = "channel"
    module.error.NetworkError = type("NetworkError", (OSError,), {})
    module.error.TimedOut = type("TimedOut", (OSError,), {})
    module.error.BadRequest = type("BadRequest", (module.error.NetworkError,), {})
    for name in (
        "telegram", "telegram.ext", "telegram.constants", "telegram.request",
    ):
        sys.modules.setdefault(name, module)
    sys.modules.setdefault("telegram.error", module.error)


_ensure_telegram_mock()

from plugins.platforms.telegram import health_actions
from plugins.platforms.telegram.adapter import TelegramAdapter


def _query(*, data: str = "hx:token:log"):
    message = SimpleNamespace(
        chat=SimpleNamespace(type="private"),
        chat_id=123,
        message_id=9,
        message_thread_id=77,
        reply_markup=object(),
    )
    return SimpleNamespace(
        id="query-1",
        data=data,
        message=message,
        from_user=SimpleNamespace(id=5, first_name="Fictional"),
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
    )


def _handler_script(path: Path) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "payload = json.loads(sys.stdin.read())\n"
        "print(json.dumps({'ok': True, 'answer_text': payload['data']}))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


@pytest.mark.asyncio
async def test_missing_worker_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "HERMES_TELEGRAM_ACTION_HANDLER", str(tmp_path / "not-present")
    )

    assert await health_actions._run("callback", {"data": "hx:test"}) == {
        "ok": False,
        "answer_text": "Tap actions are temporarily unavailable.",
    }


@pytest.mark.asyncio
async def test_worker_receives_exact_json_payload(monkeypatch, tmp_path):
    worker = _handler_script(tmp_path / "worker")
    monkeypatch.setenv("HERMES_TELEGRAM_ACTION_HANDLER", str(worker))

    result = await health_actions._run("callback", {"data": "hx:test"})

    assert result == {"ok": True, "answer_text": "hx:test"}


@pytest.mark.asyncio
async def test_unauthorized_callback_never_dispatches(monkeypatch):
    adapter = MagicMock()
    adapter._is_callback_user_authorized.return_value = False
    query = _query()
    dispatch = AsyncMock()
    monkeypatch.setattr(health_actions, "_run", dispatch)

    await health_actions.handle_callback(
        adapter, SimpleNamespace(callback_query=query)
    )

    dispatch.assert_not_awaited()
    query.answer.assert_awaited_once_with(
        text="⛔ You are not authorized to use this action."
    )


@pytest.mark.asyncio
async def test_authorized_callback_dispatches_bounded_payload(monkeypatch):
    adapter = MagicMock()
    adapter._is_callback_user_authorized.return_value = True
    query = _query()
    dispatch = AsyncMock(return_value={
        "ok": True,
        "answer_text": "Logged.",
        "edit_text": "Updated card",
        "remove_keyboard": True,
    })
    monkeypatch.setattr(health_actions, "_run", dispatch)

    await health_actions.handle_callback(
        adapter, SimpleNamespace(callback_query=query)
    )

    dispatch.assert_awaited_once_with(
        "callback",
        {
            "query_id": "query-1",
            "data": "hx:token:log",
            "chat_id": "123",
            "message_id": "9",
            "user_id": "5",
        },
    )
    query.answer.assert_awaited_once_with(text="Logged.")
    query.edit_message_text.assert_awaited_once_with(
        text="Updated card", reply_markup=None
    )


def test_register_is_disabled_without_root_owned_worker(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "HERMES_TELEGRAM_ACTION_HANDLER", str(tmp_path / "not-present")
    )
    application = MagicMock()

    assert health_actions.register(MagicMock(), application) is False
    application.add_handler.assert_not_called()


def test_register_adds_reaction_handler_when_worker_exists(
    monkeypatch, tmp_path,
):
    worker = _handler_script(tmp_path / "worker")
    monkeypatch.setenv("HERMES_TELEGRAM_ACTION_HANDLER", str(worker))
    reaction_handler = MagicMock(return_value="registered-reaction-handler")
    monkeypatch.setattr(sys.modules["telegram.ext"], "MessageReactionHandler", reaction_handler)
    application = MagicMock()
    adapter = MagicMock()

    assert health_actions.register(adapter, application) is True
    application.add_handler.assert_called_once_with("registered-reaction-handler")
    callback = reaction_handler.call_args.args[0]
    assert callable(callback)


@pytest.mark.asyncio
async def test_adapter_routes_only_hx_callback_to_health_bridge(monkeypatch):
    callback = AsyncMock()
    monkeypatch.setattr(health_actions, "handle_callback", callback)
    adapter = object.__new__(TelegramAdapter)
    update = SimpleNamespace(callback_query=_query())

    await TelegramAdapter._handle_callback_query(adapter, update, None)

    callback.assert_awaited_once_with(adapter, update)


def test_live_module_contains_no_health_reasoning_or_storage_surface():
    source = Path(health_actions.__file__).read_text(encoding="utf-8")
    assert "synthesis-record" not in source
    assert "HEALTH_DB" not in source
    assert "openrouter" not in source.lower()
    assert "requests." not in source
