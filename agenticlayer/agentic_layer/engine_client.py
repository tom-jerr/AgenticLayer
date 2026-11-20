"""Async client for the scheduler-owned, epoch-scoped control socket."""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .protocol import (
    MAX_LINE, PROTOCOL_VERSION, EngineCommand, EngineReply, WireError,
    decode_line, encode_line, identifier,
)


class EngineUnavailable(WireError):
    def __init__(self, message: str):
        super().__init__(503, "engine_unavailable", message)


class EngineClient:
    def __init__(self, socket: Path, *, timeout: float = 2.0):
        self.socket = socket
        self.timeout = timeout
        self.epoch: str | None = None
        self._failed_epoch: str | None = None
        self.capabilities: dict[str, Any] = {}
        self.snapshot: EngineReply | None = None
        self.snapshot_ns = 0
        self._cursor = 0
        self._lock = asyncio.Lock()
        self._pending: dict[str, asyncio.Future[EngineReply]] = {}
        self.on_disconnect: Callable[[], None] | None = None
        self.on_receipt: Callable[[EngineReply], None] | None = None

    def watch(self, action_id: str) -> asyncio.Future[EngineReply]:
        if action_id in self._pending:
            raise RuntimeError(f"already watching {action_id}")
        future = asyncio.get_running_loop().create_future()
        self._pending[action_id] = future
        return future

    def unwatch(self, action_id: str) -> None:
        future = self._pending.pop(action_id, None)
        if future is not None and not future.done():
            future.cancel()

    def _disconnect(self) -> None:
        was_connected = self.epoch is not None
        if self.epoch is not None:
            self._failed_epoch = self.epoch
        self.epoch = self.snapshot = None
        self.snapshot_ns = self._cursor = 0
        self.capabilities = {}
        for action_id, pending in self._pending.items():
            if not pending.done():
                pending.set_result({"status": "error", "error": "engine epoch lost",
                                    "action_id": action_id})  # type: ignore[typeddict-item]
        self._pending.clear()
        if was_connected and self.on_disconnect is not None:
            self.on_disconnect()

    async def _exchange(self, ops: list[EngineCommand]) -> list[EngineReply]:
        writer = None
        try:
            async with asyncio.timeout(self.timeout):
                reader, writer = await asyncio.open_unix_connection(self.socket, limit=MAX_LINE)
                writer.write(encode_line({"epoch": self.epoch, "ops": ops}))
                await writer.drain()
                response = decode_line(await reader.readline())
                epoch = identifier(response.get("epoch"), "epoch")
                if epoch == self._failed_epoch:
                    raise EngineUnavailable("scheduler must restart after losing control of its previous epoch")
                replies = response["replies"]
                if self.epoch is not None and self.epoch != epoch:
                    raise EngineUnavailable("scheduler epoch changed; request was not retried")
                if not isinstance(replies, list) or len(replies) != (len(ops) or 1):
                    raise ValueError("unexpected scheduler reply count")
                if any(not isinstance(r, dict) or r.get("epoch") != epoch for r in replies):
                    raise ValueError("invalid scheduler reply epoch")
                for command, reply in zip(ops, replies):
                    if reply.get("action_id") != command["action_id"]:
                        raise ValueError("scheduler reply action mismatch")
                if self.epoch is None:
                    details = replies[0].get("details", {})
                    caps = details.get("capabilities")
                    if details.get("protocol") != PROTOCOL_VERSION or not isinstance(caps, dict):
                        raise ValueError("scheduler did not advertise AgenticLayer protocol")
                    self.epoch = epoch
                    self.capabilities = caps
                return replies
        except (OSError, TimeoutError, ValueError, KeyError) as exc:
            self._disconnect()
            if isinstance(exc, EngineUnavailable):
                raise
            raise EngineUnavailable(f"scheduler control unavailable: {exc}") from exc
        finally:
            if writer is not None:
                writer.close()

    async def rpc(self, ops: list[EngineCommand]) -> list[EngineReply]:
        async with self._lock:
            if self.epoch is None:
                raise EngineUnavailable("scheduler has not completed its handshake")
            return await self._exchange(ops)

    async def poll(self) -> None:
        async with self._lock:
            command: EngineCommand = {
                "op": "snapshot", "action_id": "snapshot", "rid": "snapshot",
                "run_id": "control", "version_id": "snapshot", "handle": None,
                "payload": {"after_seq": self._cursor},
            }
            reply, = await self._exchange([command] if self.epoch else [])
            if reply.get("status") not in {"ok", "ready", "completed"}:
                self._disconnect()
                raise EngineUnavailable("scheduler rejected its snapshot")
            details = reply.get("details", {})
            for receipt in details.get("completions", []):
                seq = receipt["seq"]
                if seq <= self._cursor:
                    continue
                self._cursor = seq
                action_id = receipt["action_id"]
                future = self._pending.get(action_id)
                if future is not None and not future.done():
                    future.set_result(receipt)
                if self.on_receipt is not None:
                    self.on_receipt(receipt)
            self.snapshot = reply
            self.snapshot_ns = time.monotonic_ns()

    async def handshake(self) -> None:
        deadline = asyncio.get_running_loop().time() + self.timeout
        while True:
            try:
                async with asyncio.timeout_at(deadline):
                    await self.poll()
                return
            except (EngineUnavailable, TimeoutError):
                if asyncio.get_running_loop().time() >= deadline:
                    raise EngineUnavailable("no scheduler capability handshake within 2 seconds") from None
                await asyncio.sleep(0.025)
