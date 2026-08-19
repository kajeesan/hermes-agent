"""Opt-in cross-repository acceptance for fictional OHA Telegram delivery.

Set ``OPENHEALTHATLAS_CHECKOUT`` to the exact OpenHealthAtlas source checkout.
The harness builds a temporary schema-v5 fictional database and private
ancestry sidecar, executes the real OpenHealthAtlas adapter and deterministic
Recovery engine, projects the result through production ``RecoveryJourney``,
and feeds the exact MCP completion to Hermes' production collector. Only MCP
stdio, receipt persistence, and the final Telegram Bot API call are replaced.
"""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shutil
import sqlite3
import stat
import subprocess
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest


def _ensure_telegram_mock() -> None:
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
    FAIL_CLOSED_TEXT,
    _reset_required_delivery_state_for_tests,
    prepare_gateway_delivery,
)
from hermes_cli import plugins as plugin_module
from hermes_cli.plugins import PluginManager, PluginManifest
from model_tools import _emit_post_tool_call_hook
from plugins.platforms.telegram.adapter import TelegramAdapter


# Exact observed registry names from the accepted fictional MCP server. No
# suffix inference or nearby alias is accepted by the delivery plugin.
SNAPSHOT_TOOL = "mcp_openhealthatlas_fictional_openhealthatlas_recovery_snapshot"
DETAIL_TOOL = "mcp_openhealthatlas_fictional_openhealthatlas_recovery_detail"
GENERIC_QUERY_TOOL = "mcp_openhealthatlas_fictional_openhealthatlas_health_query"
GENERIC_EVIDENCE_TOOL = "mcp_openhealthatlas_fictional_openhealthatlas_health_evidence"
NON_FICTIONAL_ALIAS = "mcp_openhealthatlas_openhealthatlas_recovery_snapshot"
RAW_CANARY = "SECRET RAW FICTIONAL SORENESS TEXT"
CHAT_ID = "-100424242"
TOPIC_ID = "77"
RANGE_FROM = "2026-03-02"
ANCHOR = "2026-06-30"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oha_root() -> Path:
    raw = os.environ.get("OPENHEALTHATLAS_CHECKOUT", "").strip()
    if not raw:
        pytest.skip("set OPENHEALTHATLAS_CHECKOUT for cross-repository acceptance")
    root = Path(raw)
    if not root.is_absolute():
        pytest.fail("OPENHEALTHATLAS_CHECKOUT must be an absolute path")
    required = (
        root / "deploy" / "hermes-openhealthatlas-mcp",
        root / "deploy" / "hermes-openhealthatlas-tool",
        root / "deploy" / "hermes-openhealthatlas-tool-config.json",
        root / "deploy" / "hermes-ssh-transport",
        root / "deploy" / "hermes-openhealthatlas-delivery-plugin" / "__init__.py",
        root / "deploy" / "openhealthatlas-real-data-governance.json",
        root / "scripts" / "build_readiness_ancestry.py",
        root / "tests" / "test_readiness_ancestry.py",
        root / "toolkit" / "health.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        pytest.fail(f"OpenHealthAtlas checkout is missing required files: {missing}")
    return root


def _mcp_wrapper(snapshot: dict) -> str:
    return json.dumps(
        {
            "result": json.dumps(snapshot, sort_keys=True),
            "structuredContent": snapshot,
        },
        sort_keys=True,
    )


def _emit(snapshot: dict, *, session: str, turn: str, call: str, tool: str) -> None:
    _emit_post_tool_call_hook(
        function_name=tool,
        function_args={},
        result=_mcp_wrapper(snapshot),
        status="ok",
        session_id=session,
        turn_id=turn,
        tool_call_id=call,
        api_request_id=f"{turn}:request",
        task_id=f"{turn}:task",
    )


def _adapter() -> TelegramAdapter:
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="fictional-token"))
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _measurement(path: Path) -> dict:
    before = path.stat(follow_symlinks=False)
    assert stat.S_ISREG(before.st_mode) and not path.is_symlink()
    digest = _file_sha256(path)
    after = path.stat(follow_symlinks=False)

    def fields(value):
        return {
            "device": value.st_dev,
            "inode": value.st_ino,
            "mode": stat.S_IMODE(value.st_mode),
            "size": value.st_size,
            "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns,
            "uid": value.st_uid,
            "gid": value.st_gid,
        }

    assert fields(before) == fields(after)
    return {"stat": fields(before), "sha256": digest}


def _sqlite_companions(database: Path) -> list[Path]:
    return [
        Path(str(database) + suffix)
        for suffix in ("-wal", "-shm", "-journal")
        if Path(str(database) + suffix).exists()
    ]


def _seed_v5_database(oha: Path, tmp_path: Path) -> Path:
    helpers = _load_module(
        oha / "tests" / "test_readiness_ancestry.py",
        "oha_cross_repository_readiness_fixture",
    )
    database = helpers._v5_database(
        tmp_path / "fictional-recovery-v5.db", note=RAW_CANARY
    )
    anchor = date.fromisoformat(ANCHOR)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE sleep_log SET time_asleep_hours=?, quality=? WHERE date=?",
            (7.09, 5, ANCHOR),
        )
        connection.execute(
            "INSERT INTO daily_metrics(date,source,resting_hr,hrv_ms,sleep_hours) "
            "VALUES(?,?,?,?,?)",
            (ANCHOR, "fitbit", 61.0, 45.0, None),
        )
        connection.executemany(
            "INSERT INTO daily_metrics(date,source,resting_hr,hrv_ms,sleep_hours) "
            "VALUES(?,?,?,?,?)",
            [
                (
                    (anchor - timedelta(days=offset)).isoformat(),
                    "fitbit",
                    60.0,
                    50.0,
                    None,
                )
                for offset in range(1, 11)
            ],
        )
        connection.executemany(
            "INSERT INTO daily_metrics(date,source,resting_hr,hrv_ms,sleep_hours) "
            "VALUES(?,?,?,?,?)",
            [
                (
                    (anchor - timedelta(days=offset)).isoformat(),
                    "apple-health",
                    59.0,
                    52.0,
                    None,
                )
                for offset in range(1, 87)
            ],
        )
        connection.commit()
    assert _sqlite_companions(database) == []
    return database


def _build_private_sidecar(oha: Path, database: Path, sidecar: Path) -> dict:
    completed = subprocess.run(
        [
            sys.executable,
            str(oha / "scripts" / "build_readiness_ancestry.py"),
            "--database",
            str(database),
            "--from",
            RANGE_FROM,
            "--anchor",
            ANCHOR,
            "--output",
            str(sidecar),
        ],
        cwd=oha,
        env={
            **os.environ,
            "HERMES_TIMEZONE": "UTC",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == {
        "ok": True,
        "contract": "openhealthatlas-readiness-ancestry-v2",
        "fixture_lane": "accepted-v5",
        "schema_version": 5,
        "scope": "current_snapshot_integrity_and_reproducibility",
    }
    assert stat.S_IMODE(sidecar.stat().st_mode) == 0o600
    return json.loads(sidecar.read_text(encoding="utf-8"))


def _tool_config(
    *, oha: Path, tmp_path: Path, database: Path, sidecar: Path
) -> dict:
    vault = tmp_path / "fictional-vault"
    vault.mkdir()
    return {
        "audit": str(tmp_path / "adapter-audit.jsonl"),
        "contract": "openhealthatlas-hermes-tool-v1",
        "data_class": "fictional",
        "data_dir": str(tmp_path),
        "fixture_id": "comprehensive-persona-v1",
        "health_cli": str(oha / "toolkit" / "health.py"),
        "health_db": str(database),
        "health_vault": str(vault),
        "range_from": RANGE_FROM,
        "range_to": ANCHOR,
        "readiness_ancestry_sidecar": str(sidecar),
        "readiness_fixture_lane": "accepted-v5",
        "synthesis_writeback_enabled": False,
        "timeout_seconds": 30,
        "timezone": "UTC",
    }


def _load_production_mcp(
    *, oha: Path, tmp_path: Path, monkeypatch, config: dict
) -> tuple[dict, Path, tuple[Path, Path]]:
    stage = tmp_path / "staged-openhealthatlas" / "deploy"
    stage.mkdir(parents=True)
    for relative in (
        "hermes-openhealthatlas-tool",
        "hermes-openhealthatlas-tool-config.json",
        "hermes-ssh-transport",
        "openhealthatlas-real-data-governance.json",
    ):
        shutil.copy2(oha / "deploy" / relative, stage / relative)
    assert _file_sha256(stage / "hermes-openhealthatlas-tool") == _file_sha256(
        oha / "deploy" / "hermes-openhealthatlas-tool"
    )
    (stage / "tool-config.json").write_text(
        json.dumps(config, sort_keys=True), encoding="utf-8"
    )

    # Production startup chdirs to its receipt directory. Redirect only those
    # two persistence locations into this test's staging root; no Recovery
    # projection or adapter execution code is changed.
    mcp_source = (oha / "deploy" / "hermes-openhealthatlas-mcp").read_text(
        encoding="utf-8"
    )
    receipt = stage / "last-hermes-receipt.json"
    recovery_receipt = stage / "last-recovery-receipt.json"
    replacements = {
        "/var/lib/openhealthatlas-fictional/last-hermes-receipt.json": receipt,
        "/var/lib/openhealthatlas-fictional/last-recovery-receipt.json": (
            recovery_receipt
        ),
    }
    for original, replacement in replacements.items():
        quoted = json.dumps(original)
        assert mcp_source.count(quoted) == 1
        mcp_source = mcp_source.replace(quoted, json.dumps(str(replacement)))
    staged_mcp = stage / "hermes-openhealthatlas-mcp"
    staged_mcp.write_text(mcp_source, encoding="utf-8")

    class FakeFastMCP:
        def __init__(self, *_args, **_kwargs):
            pass

        @staticmethod
        def tool(*_args, **_kwargs):
            return lambda function: function

        def run(self, *_args, **_kwargs):
            raise AssertionError("acceptance must not start FastMCP stdio")

    mcp_module = types.ModuleType("mcp")
    server_module = types.ModuleType("mcp.server")
    fastmcp_module = types.ModuleType("mcp.server.fastmcp")
    fastmcp_module.FastMCP = FakeFastMCP
    monkeypatch.setitem(sys.modules, "mcp", mcp_module)
    monkeypatch.setitem(sys.modules, "mcp.server", server_module)
    monkeypatch.setitem(sys.modules, "mcp.server.fastmcp", fastmcp_module)

    previous_cwd = Path.cwd()
    try:
        namespace = runpy.run_path(str(staged_mcp))
    finally:
        os.chdir(previous_cwd)
    return namespace, stage, (receipt, recovery_receipt)


def _private_values(
    *, database: Path, sidecar_path: Path, sidecar: dict
) -> set[str]:
    snapshot = sidecar["snapshot"]
    values = {
        str(database),
        str(sidecar_path),
        sidecar["sidecar_sha256"],
        snapshot["database_sha256"],
        snapshot["canonical_recovery_rows_sha256"],
        "sha256:" + hashlib.sha256(RAW_CANARY.encode("utf-8")).hexdigest(),
    }
    for manifest in snapshot["table_manifests"]:
        values.add(manifest["rows_sha256"])
    return values


def _assert_private_absent(serialized: str, private_values: set[str]) -> None:
    assert RAW_CANARY not in serialized
    for value in private_values:
        assert value not in serialized


def test_generic_query_obligation_final_evidence_and_renderer(
    tmp_path, monkeypatch,
):
    oha = _oha_root()
    database = tmp_path / "generic-fictional.db"
    generated = subprocess.run(
        [
            sys.executable,
            str(oha / "scripts" / "make_demo_db.py"),
            "--output",
            str(database),
            "--anchor-date",
            ANCHOR,
        ],
        cwd=oha,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(generated.stdout)["data_class"] == "fictional"
    config = _tool_config(
        oha=oha,
        tmp_path=tmp_path,
        database=database,
        sidecar=tmp_path / "unused-sidecar.json",
    )
    config["readiness_fixture_lane"] = "development-v6"
    config["readiness_ancestry_sidecar"] = None
    namespace, stage, receipt_paths = _load_production_mcp(
        oha=oha, tmp_path=tmp_path, monkeypatch=monkeypatch, config=config,
    )
    monkeypatch.setenv("OPENHEALTHATLAS_SOURCE_TEST", "1")
    monkeypatch.setenv("OPENHEALTHATLAS_TEST_CONFIG", str(stage / "tool-config.json"))
    before = _measurement(database)

    query = namespace["openhealthatlas_health_query"](
        ["sleep.duration_hours", "nutrition.logged.protein_g"],
        {"from": "2026-06-01", "to": ANCHOR},
        "summary",
    )
    evidence = namespace["openhealthatlas_health_evidence"](
        [query["evidence_refs"][0]], "summary",
    )
    assert _measurement(database) == before
    assert _sqlite_companions(database) == []
    assert all(not path.exists() for path in receipt_paths)
    assert query["delivery_contract"]["completion_role"] == "requires_final"
    assert evidence["delivery_contract"]["completion_role"] == "final"
    assert evidence["coverage"]["verified_references"] == 1

    manager = PluginManager()
    manifest = PluginManifest(
        name="openhealthatlas-recovery-delivery",
        key="openhealthatlas-recovery-delivery",
        version="1.0.0",
        source="user",
        path=str(oha / "deploy" / "hermes-openhealthatlas-delivery-plugin"),
    )
    manager._load_plugin(manifest)
    assert manager._plugins[manifest.key].enabled is True
    monkeypatch.setattr(plugin_module, "_plugin_manager", manager)
    _reset_required_delivery_state_for_tests()
    assert manager.get_gateway_delivery_requirements_for_tool(GENERIC_QUERY_TOOL) == [
        ("openhealthatlas-generic-evidence-delivery", "1.0.1")
    ]
    assert manager.get_gateway_delivery_requirements_for_tool(GENERIC_EVIDENCE_TOOL) == [
        ("openhealthatlas-generic-evidence-delivery", "1.0.1")
    ]

    _emit(
        query,
        session="generic-session",
        turn="generic-turn",
        call="query-call",
        tool=GENERIC_QUERY_TOOL,
    )
    _emit(
        evidence,
        session="generic-session",
        turn="generic-turn",
        call="evidence-call",
        tool=GENERIC_EVIDENCE_TOOL,
    )
    prepared = prepare_gateway_delivery(
        session_id="generic-session",
        turn_id="generic-turn",
        response_text=(
            "Deterministic result\nThe bounded fictional sleep and nutrition "
            "summaries are available for interpretation."
        ),
        platform="telegram",
        destination_id=CHAT_ID,
        destination_topic_id=TOPIC_ID,
    )
    assert prepared is not None
    assert str(prepared).startswith("Data & evidence disclosure\n")
    assert "Deterministic result" in str(prepared)
    assert "sleep.duration_hours" in str(prepared)
    assert "nutrition.logged.protein_g" in str(prepared)
    assert "Hermes interpretation (model-generated" in str(prepared)


@pytest.mark.asyncio
async def test_actual_fictional_snapshot_plugin_collector_and_telegram_boundary(
    tmp_path, monkeypatch,
):
    oha = _oha_root()
    database = _seed_v5_database(oha, tmp_path)
    sidecar_path = tmp_path / "private-readiness-ancestry.json"
    sidecar = _build_private_sidecar(oha, database, sidecar_path)
    config = _tool_config(
        oha=oha,
        tmp_path=tmp_path,
        database=database,
        sidecar=sidecar_path,
    )
    namespace, stage, receipt_paths = _load_production_mcp(
        oha=oha, tmp_path=tmp_path, monkeypatch=monkeypatch, config=config
    )
    monkeypatch.setenv("OPENHEALTHATLAS_SOURCE_TEST", "1")
    monkeypatch.setenv(
        "OPENHEALTHATLAS_TEST_CONFIG", str(stage / "tool-config.json")
    )

    database_before = _measurement(database)
    sidecar_before = _measurement(sidecar_path)
    assert _sqlite_companions(database) == []

    receipt_writes = []
    journey = namespace["RecoveryJourney"]()
    journey._save_receipt = lambda focus=None: receipt_writes.append(focus)
    snapshot = journey.snapshot()

    assert _measurement(database) == database_before
    assert _measurement(sidecar_path) == sidecar_before
    assert _sqlite_companions(database) == []
    assert receipt_writes == [None]
    assert all(not path.exists() for path in receipt_paths)

    audit_path = Path(config["audit"])
    audit_rows = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["phase"] for row in audit_rows] == ["intent", "result"]
    assert all(row["command"][2] == "readiness" for row in audit_rows)
    assert all("synthesis-record" not in row["command"] for row in audit_rows)
    assert audit_rows[0]["database_before_sha256"] == database_before["sha256"]
    assert audit_rows[1]["database_before_sha256"] == database_before["sha256"]
    assert audit_rows[1]["database_after_sha256"] == database_before["sha256"]
    assert audit_rows[1]["returncode"] == 0

    assert snapshot["result"]["status"] == "insufficient_data"
    assert snapshot["result"]["components"] == [
        {
            "key": "sleep",
            "score": 92,
            "value": 7.09,
            "basis": "7.1h · quality 5/5 vs 8h target",
        }
    ]
    evidence_by_key = {
        row["key"]: row for row in snapshot["result"]["evidence"]["components"]
    }
    warning_by_key = {
        row["component"]: row
        for row in snapshot["result"]["evidence"]["warnings"]
    }
    assert evidence_by_key["sleep"]["status"] == "included"
    for key in ("hrv", "rhr"):
        assert evidence_by_key[key]["status"] == "excluded"
        assert (
            evidence_by_key[key]["reason_code"]
            == "insufficient_same_source_baseline"
        )
        assert evidence_by_key[key]["baseline"]["source_label"] == "fitbit"
        assert evidence_by_key[key]["baseline"]["observation_count"] == 10
        assert warning_by_key[key] == {
            "code": "insufficient_same_source_baseline",
            "component": key,
            "source_label": "fitbit",
            "required": 14,
            "observed": 10,
            "other_source_observations_excluded": 86,
        }
    assert snapshot["result"]["evidence"]["snapshot_integrity"] == {
        "contract": "openhealthatlas-readiness-ancestry-v2",
        "status": "verified",
        "scope": "current_snapshot_integrity_and_reproducibility",
        "schema_version": 5,
        "external_provider_sync": "not_performed",
        "training_ancestry": "manifested",
    }
    assert snapshot["required_disclosure"]["interpretation_writeback"] == "disabled"
    assert snapshot["sharing_authorization"]["interpretation_writeback"] == "disabled"
    assert snapshot["sharing_authorization"]["calculation_fingerprint_member"] is False
    assert config["synthesis_writeback_enabled"] is False

    private_values = _private_values(
        database=database, sidecar_path=sidecar_path, sidecar=sidecar
    )
    serialized_snapshot = json.dumps(snapshot, sort_keys=True)
    _assert_private_absent(serialized_snapshot, private_values)

    manager = PluginManager()
    manifest = PluginManifest(
        name="openhealthatlas-recovery-delivery",
        key="openhealthatlas-recovery-delivery",
        version="1.0.0",
        source="user",
        path=str(oha / "deploy" / "hermes-openhealthatlas-delivery-plugin"),
    )
    manager._load_plugin(manifest)
    loaded = manager._plugins[manifest.key]
    assert loaded.enabled is True, loaded.error
    monkeypatch.setattr(plugin_module, "_plugin_manager", manager)
    _reset_required_delivery_state_for_tests()

    requirements = manager.get_gateway_delivery_requirements_for_tool(SNAPSHOT_TOOL)
    assert requirements == [("openhealthatlas-recovery-delivery", "1.0.0")]
    assert manager.get_gateway_delivery_requirements_for_tool(DETAIL_TOOL) == requirements
    assert manager.get_gateway_delivery_requirements_for_tool(NON_FICTIONAL_ALIAS) == []

    _emit(
        snapshot,
        session="alias-session",
        turn="alias-turn",
        call="alias-call",
        tool=NON_FICTIONAL_ALIAS,
    )
    alias_payload = prepare_gateway_delivery(
        session_id="alias-session",
        turn_id="alias-turn",
        response_text="must not survive",
        platform="telegram",
        destination_id=CHAT_ID,
        destination_topic_id=TOPIC_ID,
    )
    assert str(alias_payload) == FAIL_CLOSED_TEXT
    assert "must not survive" not in str(alias_payload)

    _emit(
        snapshot, session="session", turn="turn", call="call", tool=SNAPSHOT_TOOL
    )
    assert prepare_gateway_delivery(
        session_id="wrong-session",
        turn_id="turn",
        response_text="optional",
        platform="telegram",
        destination_id=CHAT_ID,
        destination_topic_id=TOPIC_ID,
    ) is None
    assert prepare_gateway_delivery(
        session_id="session",
        turn_id="wrong-turn",
        response_text="optional",
        platform="telegram",
        destination_id=CHAT_ID,
        destination_topic_id=TOPIC_ID,
    ) is None
    optional_prose = "The deterministic result is limited by the stated evidence"
    prepared = prepare_gateway_delivery(
        session_id="session",
        turn_id="turn",
        response_text=optional_prose,
        platform="telegram",
        destination_id=CHAT_ID,
        destination_topic_id=TOPIC_ID,
    )
    assert prepared is not None
    assert str(prepared).startswith("Data & evidence disclosure\n")
    assert "insufficient_data" in str(prepared)
    assert "Hermes interpretation\n" in str(prepared)
    assert str(prepared).endswith(optional_prose)
    _assert_private_absent(str(prepared), private_values)

    adapter = _adapter()
    sent = []

    async def send_message(**kwargs):
        sent.append(kwargs)
        result = MagicMock()
        result.message_id = len(sent)
        return result

    adapter._bot.send_message = AsyncMock(side_effect=send_message)
    metadata = {
        "required_delivery_token": prepared.required_delivery_token,
        "thread_id": TOPIC_ID,
        "notify": True,
    }

    redirected_chat = await adapter.send(
        chat_id="-100424243", content=str(prepared), metadata=metadata
    )
    redirected_topic = await adapter.send(
        chat_id=CHAT_ID,
        content=str(prepared),
        metadata={**metadata, "thread_id": "78"},
    )
    assert redirected_chat.success is False
    assert redirected_topic.success is False
    adapter._bot.send_message.assert_not_awaited()

    delivered = await adapter._send_with_retry(
        chat_id=CHAT_ID,
        content=prepared,
        metadata=metadata,
        base_delay=0,
    )
    assert delivered.success is True
    assert sent
    assert sent[0]["text"].startswith("Data & evidence disclosure")
    assert "Hermes interpretation" in sent[-1]["text"]
    assert optional_prose in sent[-1]["text"]
    assert sum(optional_prose in item["text"] for item in sent) == 1
    assert all(item["message_thread_id"] == int(TOPIC_ID) for item in sent)
    public_delivery = json.dumps(sent, default=str, sort_keys=True)
    _assert_private_absent(public_delivery, private_values)

    calls_after_success = adapter._bot.send_message.await_count
    replay = await adapter.send(
        chat_id=CHAT_ID, content=prepared, metadata=metadata
    )
    assert replay.success is False
    assert adapter._bot.send_message.await_count == calls_after_success
