"""Loopback-only ASGI ingress, version directory and control executor."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import time
import uuid
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .engine_client import EngineClient, EngineUnavailable
from .policy import Calibration, Candidate, Node, Policy, Snapshot
from .protocol import (
    BRANCH_HEADER, MAX_BATCH, MAX_CONTEXT, MAX_GENERATE, MAX_REPLAY_GENERATE,
    REQUEST_HEADER, SESSION_HEADER, VERSION_HEADER, EngineCommand, EngineReply, Event,
    EventKind, WireError, conflict, identifier, invalid, object_value, token_ids, unknown,
)
from .versions import Action, Branch, Directory, PERMIT_NS, TERMINAL, Version, digest

LOG = logging.getLogger("agentic_layer")
MODEL = "qwen35-9b-nvfp4"
TICK = 0.025
MAX_BODY = 8 << 20
T = TypeVar("T")
_BIND_OK = {"ok", "accepted", "ready"}
_DONE = {"ok", "completed", "ready", "partial"}
_SERIAL = {"baseline", "lifecycle-only", "offload-only", "lookahead-only", "ttl", "task-prefix"}
_NO_PLACEMENT = (_SERIAL - {"offload-only"}) | {"parallel-only"}


@dataclass(frozen=True, slots=True)
class Settings:
    bearer: str
    socket: Path
    tokenizer_hash: str
    layout_hash: str
    backend: str = "http://127.0.0.1:30080"
    model_revision: str = "97aef92393f126bf649f310cd40861be8dad3279"
    mode: str = "strong"
    calibration: Path | None = None
    disabled_mechanisms: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        url = urlsplit(self.backend)
        if (url.scheme != "http" or url.hostname != "127.0.0.1" or url.path not in {"", "/"}
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("SGLang backend must be a 127.0.0.1 HTTP origin")
        if not self.bearer or not self.socket.is_absolute() or not self.tokenizer_hash or not self.layout_hash:
            raise ValueError("bearer, absolute engine socket and runtime fingerprints are required")

    @classmethod
    def from_env(cls) -> Settings:
        calibration = os.getenv("AGENTIC_CALIBRATION")
        return cls(os.environ["AGENTIC_LOCAL_TOKEN"], Path(os.environ["AGENTIC_ENGINE_SOCKET"]),
                   os.environ["AGENTIC_TOKENIZER_HASH"], os.environ["AGENTIC_LAYOUT_HASH"],
                   os.getenv("AGENTIC_BACKEND", "http://127.0.0.1:30080"),
                   mode=os.getenv("AGENTIC_POLICY", "strong"),
                   calibration=Path(calibration) if calibration else None,
                   disabled_mechanisms=frozenset(os.getenv("AGENTIC_DISABLE_MECHANISMS", "").split()))


@dataclass(slots=True)
class Dispatch:
    action: Action
    branch: Branch
    version: Version
    epoch: str
    expected: tuple[int, ...]
    salt: str
    future: asyncio.Future[EngineReply]
    finished: asyncio.Future[EngineReply]
    aborting: bool = False


@dataclass(slots=True)
class Movement:
    candidate: Candidate
    branch: Branch
    version: Version
    op: str
    done: asyncio.Event = field(default_factory=asyncio.Event)


def _context_size(length: int, budget: int) -> None:
    if not length:
        raise invalid("prompt is empty")
    if length + budget > MAX_CONTEXT:
        raise WireError(400, "context_length_exceeded",
                        f"This model's maximum context length is {MAX_CONTEXT} tokens. "
                        f"The request needs {length} prompt tokens and {budget} completion tokens.")


def _context(ids: tuple[int, ...], budget: int) -> None:
    _context_size(len(ids), budget)


def _budget(value: Any, cap: int) -> int:
    if type(value) is not int or not 1 <= value <= cap:
        raise invalid(f"generation budget must be an integer in [1, {cap}]")
    return value


async def _body(request: Request) -> Any:
    raw = bytearray()
    async for data in request.stream():
        raw.extend(data)
        if len(raw) > MAX_BODY:
            raise WireError(413, "request_too_large", "request body exceeds 8 MiB")
    try:
        return json.loads(raw)
    except (ValueError, RecursionError) as exc:
        raise invalid("invalid JSON body") from exc


async def _connected(request: Request, call: Awaitable[T], cancel: Callable[[], None]) -> T:
    async def disconnected() -> None:
        while not await request.is_disconnected():
            await asyncio.sleep(TICK)

    task = asyncio.ensure_future(call)
    monitor = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait((task, monitor), return_when=asyncio.FIRST_COMPLETED)
        if monitor in done:
            cancel()
            raise WireError(499, "client_disconnected", "request disconnected")
        return task.result()
    except asyncio.CancelledError:
        cancel()
        raise
    finally:
        monitor.cancel()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, monitor, return_exceptions=True)


class Coordinator:
    def __init__(self, settings: Settings, http: httpx.AsyncClient):
        self.http = http
        self.engine = EngineClient(settings.socket)
        self.directory = Directory(settings.model_revision, settings.tokenizer_hash, settings.layout_hash)
        calibration = Calibration.load(settings.calibration) if settings.calibration else None
        self.policy = Policy(settings.mode, calibration, disabled=settings.disabled_mechanisms)
        self.engine.on_disconnect = self.directory.lose_epoch
        self.engine.on_receipt = self.receipt
        self.wake = asyncio.Event()
        self.tasks: set[asyncio.Task[None]] = set()
        self.inflight: dict[str, Dispatch] = {}
        self.preparing: dict[str, tuple[int, ...]] = {}
        self.movements: dict[str, Movement] = {}
        self.move_cooldown: dict[str, int] = {}
        self.fork_leases: dict[tuple[str, str], tuple[Version, str, str, str]] = {}
        self.fork_pins: dict[tuple[str, str], asyncio.Task[None]] = {}
        self.retained: dict[tuple[str, str], tuple[Version, str, str]] = {}
        self.decision_log: dict[str, str] = {}

    def spawn(self, job: Awaitable[None]) -> asyncio.Task[None]:
        async def run() -> None:
            try:
                await job
            except Exception:
                LOG.exception("coordinator background action failed")
            finally:
                self.wake.set()
        task = asyncio.create_task(run())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    async def close(self) -> None:
        tasks = tuple(self.tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def start(self) -> None:
        try:
            async with asyncio.timeout(2):
                response = await self.http.get("/health")
                response.raise_for_status()
        except (TimeoutError, httpx.HTTPError) as exc:
            raise EngineUnavailable(f"native SGLang health failed: {exc}") from exc
        await self.engine.handshake()
        if not all(self.engine.capabilities.get(key) is True for key in ("bind", "prepare", "lease", "release", "completion_paging")):
            raise EngineUnavailable("scheduler does not advertise required bind/prepare/lease capabilities")
        self.spawn(self.loop())

    def receipt(self, reply: EngineReply) -> None:
        version = self.directory.versions.get(reply["version_id"])
        details = reply.get("details", {})
        if version is not None:
            residency = details.get("residency")
            if residency in ("hbm", "host", "absent"):
                version.residency = residency
            if reply.get("handle"):
                version.handle = reply["handle"]
        LOG.info("engine receipt %s", json.dumps({**reply, "details": {k: v for k, v in details.items() if k != "token_ids"}}))

    def _snapshot(self, now: int) -> Snapshot:
        raw = self.engine.snapshot or {}
        details = raw.get("details", {})
        return Snapshot(now, self.engine.snapshot_ns, details.get("hbm_capacity_bytes", 0),
                        raw.get("unique_hbm_bytes", 0), details.get("host_capacity_bytes", 0),
                        raw.get("unique_host_bytes", 0), details.get("hbm_reserved_bytes", raw.get("reserved_bytes", 0)),
                        details.get("host_reserved_bytes", 0), raw.get("fg_queue", 0), raw.get("bg_queue", 0),
                        bool(self.movements or any(d.action.kind == "prepare" for d in self.inflight.values())))

    def _model_count(self) -> int:
        return sum(a.kind in ("model", "summary", "batch") and
                   (a.status in ("granted", "running") or a.id in self.inflight)
                   for a in self.directory.actions.values())

    def _eligible(self, action: Action, branch: Branch) -> bool:
        if not self.directory.dispatchable(action, branch):
            return False
        if any(not self.directory.actions[aid].committed for aid in action.dependencies):
            return False
        if action.kind == "prepare":
            return branch.owner is None and not any(
                a.kind in {"model", "summary", "batch"} and a.status == "granted"
                for a in (self.directory.actions[aid] for aid in branch.action_ids)) and not any(
                d.branch is branch for d in self.inflight.values())
        if action.kind in ("model", "summary", "batch"):
            return branch.owner is None and self._model_count() < MAX_BATCH and not any(
                a.id != action.id and a.kind in ("model", "summary", "batch") and a.status == "granted"
                for a in (self.directory.actions[aid] for aid in branch.action_ids))
        if action.kind in ("tool", "child"):
            # A parent tool waiting on children never owns an activation slot.
            dynamic = "exclusive" if action.kind == "tool" else "exclusive_child"
            for aid in branch.action_ids:
                other = self.directory.actions[aid]
                if (other.id != action.id and other.kind == action.kind and
                        (other.status in ("granted", "running") or other.draining) and
                        (self.policy.mode in _SERIAL or "parallel" in self.policy.disabled
                         or action.payload.get(dynamic) or other.payload.get(dynamic))):
                    return False
        return True

    @staticmethod
    def checkpoint(branch: Branch) -> Version:
        version = branch.current
        while version.tokens is None and version.parent is not None:
            if version.parent is branch.inherited or version.history != version.parent.history:
                break
            version = version.parent
        return version

    def _pending(self, now: int, snapshot: Snapshot) -> tuple[list[Candidate], list[Node]]:
        actions = [a for a in self.directory.actions.values() if a.status in ("ready", "granted", "running")]
        active = {a.id for a in actions}
        calibration = self.policy.calibration
        nodes = []
        for action in actions:
            estimate = action.payload.get("service_ms")
            if type(estimate) not in (int, float) or not 0 <= estimate <= 900_000:
                estimate = 0.0
            if not estimate and calibration and action.kind in {"model", "batch", "summary", "child"}:
                current = self.checkpoint(self.directory.branches[action.key])
                if current.tokens and (bucket := calibration.bucket(len(current.tokens), 1)):
                    estimate = bucket.prefill_ms
            nodes.append(Node(action.id, estimate, tuple(dep for dep in action.dependencies if dep in active)))
        candidates = []
        for action in actions:
            branch = self.directory.branches[action.key]
            if action.deadline_ns <= now or (action.status == "granted" and action.permit is not None
                                             and action.permit["expires_mono_ns"] <= now):
                self.cancel(action, "dispatch permit or action deadline expired")
                continue
            if action.status != "ready":
                continue
            if branch.current.id != action.version_id and not self.directory.dispatchable(action, branch):
                self.cancel(action, "committed version advanced before dispatch")
                continue
            if self.engine.epoch is None:
                continue
            if action.kind == "prepare":
                ids = self.preparing.get(action.id)
                if ids is None:
                    continue
            else:
                ids = self.checkpoint(branch).tokens
            tokens = len(ids) if ids else 1
            optional = action.kind == "prepare"
            ranked = action.kind == "child" or (self.policy.mode == "task-prefix" and action.kind in ("model", "batch"))
            required = not optional and (not ranked or now - action.ready_ns >= 500_000_000
                                         or calibration is None or calibration.bucket(tokens, 1) is None)
            geometry = (self.engine.snapshot or {}).get("details", {})
            byte_stride = geometry.get("full_slot_bytes", 0)
            state_stride = geometry.get("recurrent_slot_bytes", 0)
            elapsed_ms = (now - action.ready_ns) / 1e6
            oracle_wait = action.payload.get("oracle_wait_ms") if self.policy.mode == "oracle" else None
            candidate = Candidate(action.id, action.version_id, *action.key, "prepare" if optional else "admit",
                                  action.ready_ns, action.deadline_ns, legal=self._eligible(action, branch),
                                  necessary=required, foreground=not optional, background=optional, token_count=tokens,
                                  node_id=action.id, matched_tokens=branch.current.matched,
                                  hbm_reservation_bytes=(tokens * byte_stride + state_stride) if optional and snapshot.hbm_capacity_bytes else 0,
                                  decode_hbm_bytes=MAX_GENERATE * byte_stride if action.kind == "child" and not required else 0,
                                  wait_type=action.payload.get("wait_type", ""),
                                  wait_elapsed_ms=elapsed_ms,
                                  oracle_remaining_ms=max(0.0, oracle_wait - elapsed_ms) if oracle_wait is not None else None)
            candidates.append(candidate)
        return candidates, nodes

    def _placement(self, now: int) -> dict[str, Movement]:
        result: dict[str, Movement] = {}
        if self.engine.epoch is None:
            return result
        moving = {move.version.id for move in self.movements.values()}
        details = (self.engine.snapshot or {}).get("details", {})
        slots = details.get("full_slot_bytes", 0)
        recurrent = details.get("recurrent_slot_bytes", 0)
        for branch in self.directory.branches.values():
            version = self.checkpoint(branch)
            if not version.handle or version.id in moving or not version.tokens or branch.owner is not None:
                continue
            busy = [self.directory.actions[aid] for aid in branch.action_ids
                    if self.directory.actions[aid].status in ("granted", "running") or self.directory.actions[aid].draining]
            waiting = [a for a in busy if a.draining and a.kind in ("tool", "child")]
            wait = waiting[0] if len(waiting) == 1 else None
            wait_type = (wait.payload.get("wait_type", "delegation" if wait.kind == "child" else "local_tool")
                         if wait else "delegation" if any(a.kind == "child" for a in waiting)
                         else "local_tool" if waiting else "")
            estimated = wait if self.policy.mode == "oracle" and wait and "oracle_wait_ms" in wait.payload else None
            for op, allowed, destination in (
                ("offload", (branch.lifecycle == "suspended" or branch.lifecycle == "active" and bool(waiting))
                 and version.residency == "hbm", "host"),
                ("restore", branch.lifecycle == "active" and version.residency == "host" and bool(waiting), "hbm"),
                ("reclaim", branch.lifecycle == "expired" or (self.policy.mode == "ttl" and branch.lifecycle == "suspended"), "absent"),
                ("retain", branch.lifecycle == "active" and branch.key not in self.retained and
                 version.residency == "hbm" and bool(waiting), ""),
            ):
                if not allowed or not self.engine.capabilities.get("lease" if op == "retain" else op) or version.residency == destination:
                    continue
                # A version shared by a live fork is not an expiry candidate.
                if op == "reclaim" and any(b.inherited is version and b.lifecycle != "expired"
                                           for b in self.directory.branches.values()):
                    continue
                if now - self.move_cooldown.get(f"{op}/{version.id}", 0) < 500_000_000:
                    continue
                aid = f"{op}/{version.id}"
                size = slots * len(version.tokens) + recurrent
                c = Candidate(aid, version.id, *branch.key, op, branch.last_activity_ns, now + PERMIT_NS,
                              expired=branch.lifecycle == "expired", background=op in ("offload", "restore", "retain"),
                              token_count=len(version.tokens), matched_tokens=version.matched,
                              hbm_reservation_bytes=size if op == "restore" else 0,
                              hbm_delta_bytes=-size if op == "offload" else 0,
                              host_reservation_bytes=size if op == "offload" else 0,
                              transfer_bytes=size if op in ("offload", "restore") else 0, wait_type=wait_type,
                              wait_elapsed_ms=(now - (wait.ready_ns if wait else branch.last_activity_ns)) / 1e6,
                              oracle_remaining_ms=(max(0.0, estimated.payload["oracle_wait_ms"] -
                                                       (now - estimated.ready_ns) / 1e6) if estimated else None),
                              last_migration_ns=self.move_cooldown.get(version.id),
                              suspended_since_ns=branch.last_activity_ns if branch.lifecycle == "suspended" else None)
                result[aid] = Movement(c, branch, version, op)
        return result

    async def loop(self) -> None:
        next_poll = 0
        while True:
            now = time.monotonic_ns()
            if now >= next_poll:
                with suppress(EngineUnavailable):
                    await self.engine.poll()
                next_poll = time.monotonic_ns() + int(TICK * 1e9)
            for branch in self.directory.branches.values():
                if branch.owner and branch.owner not in self.inflight and self.directory.actions[branch.owner].status in TERMINAL:
                    branch.owner = None
                lease = self.fork_leases.get(branch.key)
                if lease and lease[2] != self.engine.epoch:
                    self.fork_leases.pop(branch.key)
                elif lease and branch.lifecycle == "expired" and branch.owner is None:
                    self.fork_leases.pop(branch.key)
                    self.spawn(self.release(branch, lease[0], handle=lease[1], epoch=lease[2], residency=lease[3]))
                retained = self.retained.get(branch.key)
                if retained and (retained[2] != self.engine.epoch or branch.lifecycle != "active"
                                 or self.checkpoint(branch) is not retained[0]):
                    self.retained.pop(branch.key)
                    self.spawn(self.release(branch, retained[0], handle=retained[1], epoch=retained[2]))
            for dispatch in tuple(self.inflight.values()):
                if dispatch.action.status in {"canceled", "error"} and not dispatch.aborting:
                    dispatch.aborting = True
                    self.spawn(self.abort(dispatch))
            for branch, candidate in self.directory.discarded:
                self.preparing.pop(candidate.action_id, None)
                self.drop_candidate_lease(branch, candidate.version)
            self.directory.discarded.clear()
            for _ in range(MAX_BATCH):
                now = time.monotonic_ns()
                snapshot = self._snapshot(now)
                pending, nodes = self._pending(now, snapshot)
                placement = self._placement(now)
                selection = self.policy.choose(snapshot, [*pending, *(x.candidate for x in placement.values())], nodes)
                decision_ms = (time.monotonic_ns() - now) / 1e6 if selection.decisions else 0
                for decision in selection.decisions:
                    reason = decision.reason or ("selected" if decision.accepted else "not_selected")
                    if self.decision_log.get(decision.action_id) != reason:
                        self.decision_log[decision.action_id] = reason
                        action = self.directory.actions.get(decision.action_id)
                        movement = placement.get(decision.action_id)
                        run_id = action.run_id if action else movement.branch.run_id if movement else None
                        LOG.info("policy %s", json.dumps({"mono_ns": now, "run_id": run_id, "rid": action.rid if action else decision.action_id,
                                 "action_id": decision.action_id, "version_id": decision.version_id,
                                 "kind": decision.kind, "reason": reason, "score": decision.score,
                                 "decision_ms": decision_ms, "costs": decision.costs}))
                    if decision.kind == "prepare" and reason in {"uncalibrated", "disabled_by_mode", "missing_calibration_bucket"}:
                        action = self.directory.actions[decision.action_id]
                        self.cancel(action, reason)
                choice = selection.chosen
                if choice is None:
                    break
                if choice.action_id in placement:
                    move = placement[choice.action_id]
                    self.movements[choice.action_id] = move
                    self.spawn(self.move(move))
                    break
                action = self.directory.actions[choice.action_id]
                self.directory.grant(action, min(choice.expires_mono_ns, now + PERMIT_NS))
                if action.kind == "prepare":
                    self.spawn(self.prepare(action))
                    break
            self.wake.clear()
            with suppress(TimeoutError):
                await asyncio.wait_for(self.wake.wait(), TICK)

    def cancel(self, action: Action, reason: str) -> None:
        if action.status not in TERMINAL:
            action.set_status("canceled", reason)
        branch = self.directory.branches[action.key]
        if branch.owner == action.id and action.id not in self.inflight:
            branch.owner = None
        if branch.candidate is not None and branch.candidate.action_id == action.id:
            self.directory._discard(branch, branch.candidate)
        self.wake.set()

    async def grant(self, action: Action) -> None:
        while action.status == "ready":
            action.changed.clear()
            await action.changed.wait()

    async def tokenize(self, body: dict[str, Any]) -> tuple[int, ...]:
        native_fields = ("model", "messages", "tools", "tool_choice", "chat_template_kwargs", "reasoning_effort",
                         "continue_final_message", "parallel_tool_calls", "response_format", "thinking",
                         "add_generation_prompt")
        result = await self.http.post("/v1/tokenize", json={k: body[k] for k in native_fields if k in body})
        if result.is_error:
            raise invalid(f"native tokenizer rejected request ({result.status_code})")
        try:
            raw_ids = result.json()["tokens"]
            if isinstance(raw_ids, list) and len(raw_ids) > MAX_CONTEXT:
                _context_size(len(raw_ids), 0)
            return token_ids(raw_ids)
        except WireError as exc:
            if exc.code == "context_length_exceeded":
                raise
            raise EngineUnavailable("native tokenizer returned an invalid token sequence") from exc
        except (ValueError, KeyError) as exc:
            raise EngineUnavailable("native tokenizer returned an invalid token sequence") from exc

    @staticmethod
    def command(op: str, branch: Branch, version: Version, *, aid: str | None = None,
                rid: str | None = None, payload: dict[str, Any] | None = None) -> EngineCommand:
        aid = aid or uuid.uuid4().hex
        command: EngineCommand = {"op": op, "action_id": aid, "rid": rid or aid, "run_id": branch.run_id,
                                  "version_id": version.id, "handle": version.handle}
        if payload is not None:
            command["payload"] = payload
        return command

    def foreground_salt(self, action: Action, branch: Branch, version: Version,
                        ids: tuple[int, ...], salt: str) -> str:
        checkpoint = self.checkpoint(branch)
        if ("placement" in self.policy.disabled or self.policy.mode in _NO_PLACEMENT or action.kind == "prepare"
                or checkpoint.history != version.history or checkpoint.residency != "host"
                or not self.engine.capabilities.get("restore") or not checkpoint.tokens
                or ids[:len(checkpoint.tokens)] != checkpoint.tokens):
            return salt
        now = time.monotonic_ns()
        geometry = (self.engine.snapshot or {}).get("details", {})
        length = len(checkpoint.tokens)
        size = length * geometry.get("full_slot_bytes", 0) + geometry.get("recurrent_slot_bytes", 0)
        # Both alternatives still have to pass the scheduler's real slot gate.
        # The default with no measured estimate is native on-demand restore.
        if self.policy.calibration is None or self.policy.calibration.bucket(length, 1) is None:
            return salt
        variants = [Candidate(f"{action.id}/{kind}", version.id, *branch.key, kind,
                              action.ready_ns, action.deadline_ns, foreground=True, necessary=True,
                              token_count=length, transfer_bytes=size if kind == "restore" else 0,
                              wait_type=action.payload.get("wait_type", "local_tool"))
                    for kind in ("restore", "recompute")]
        selection = self.policy.choose(self._snapshot(now), variants)
        if selection.chosen is None:
            return salt
        LOG.info("foreground cache %s", json.dumps({"run_id": action.run_id, "rid": action.rid,
                 "action_id": action.id, "version_id": version.id, "kind": selection.chosen.kind,
                 "costs": {option.kind: dict(option.costs) for option in selection.decisions}}))
        if selection.chosen.kind == "recompute":
            version.history = digest(version.history, "on-demand-recompute", action.id)
            return self.directory.salt(branch, version, ids)
        return salt

    async def bind(self, action: Action, version: Version, ids: tuple[int, ...], budget: int) -> Dispatch:
        branch = self.directory.branches[action.key]
        if action.status != "running" or self.engine.epoch is None or not self.directory.dispatchable(action, branch):
            raise conflict("action is not dispatchable")
        if action.id in self.inflight:
            raise conflict("rid already dispatched")
        epoch = self.engine.epoch
        if action.kind != "prepare":
            prepared = next((d for d in self.inflight.values() if d.branch is branch and
                             d.action.kind == "prepare"), None)
            if prepared is not None:
                # The scheduler admits one forward per branch. Even a canceled
                # proposal owns that slot until its native terminal receipt.
                self.cancel(prepared.action, "foreground superseded active preparation")
                await self.complete(prepared)
        use = self.directory.salt(branch, version, ids)
        for move in tuple(self.movements.values()):
            if move.op == "restore" and self.directory.salt(move.branch, move.version, move.version.tokens or ()) == use:
                try:
                    await asyncio.wait_for(move.done.wait(), max(0.0, (action.deadline_ns - time.monotonic_ns()) / 1e9))
                except TimeoutError:
                    raise WireError(504, "transfer_fence_timeout", "native restore did not finish before dispatch deadline") from None
        if action.kind != "prepare" and self.retained:
            pools = (self.engine.snapshot or {}).get("details", {}).get("pools", {})
            full, recurrent = pools.get("full", {}), pools.get("recurrent", {})
            if (full.get("free", 0) - full.get("reserved", 0) < len(ids) + budget or
                    recurrent.get("free", 0) - recurrent.get("reserved", 0) < 1):
                # Optional retention must never prevent a committed request
                # from trying native eviction/admission.
                released, self.retained = self.retained, {}
                for key, (kept, handle, lease_epoch) in released.items():
                    await self.release(self.directory.branches[key], kept, handle=handle, epoch=lease_epoch)
        if action.kind == "prepare" and branch.owner is not None:
            raise conflict("foreground already owns this branch")
        if self.engine.epoch != epoch or action.status != "running" or not self.directory.dispatchable(action, branch):
            raise conflict("action changed while waiting for the native transfer fence")
        use = self.foreground_salt(action, branch, version, ids, use)
        dispatch = Dispatch(action, branch, version, epoch, ids, use, self.engine.watch(action.id),
                            asyncio.get_running_loop().create_future())
        self.inflight[action.id] = dispatch
        self.spawn(self.finish_native(dispatch))
        kind = "chat" if action.kind in ("model", "summary") else action.kind
        payload = {"expected_ids": list(ids), "cache_salt": use, "kind": kind,
                   "session_id": branch.key[0], "branch_id": branch.key[1], "max_new_tokens": budget,
                   "hold": kind == "prepare"}
        try:
            reply, = await self.engine.rpc([self.command("bind", branch, version, aid=action.id, rid=action.rid, payload=payload)])
            if reply["status"] not in _BIND_OK:
                if not dispatch.future.done():
                    dispatch.future.set_result(reply)
                raise WireError(503, "engine_resource", reply.get("error") or "scheduler denied bind")
            if action.status in TERMINAL or self.engine.epoch != epoch or not self.directory.dispatchable(action, branch):
                raise conflict("action canceled or epoch changed before HTTP dispatch")
            return dispatch
        except BaseException:
            if not dispatch.aborting:
                dispatch.aborting = True
                self.spawn(self.abort(dispatch))
            raise

    def retire(self, dispatch: Dispatch) -> None:
        self.inflight.pop(dispatch.action.id, None)
        self.engine.unwatch(dispatch.action.id)
        self.wake.set()

    async def abort(self, dispatch: Dispatch) -> None:
        if self.engine.epoch != dispatch.epoch:
            return
        try:
            await self.engine.rpc([self.command("cancel", dispatch.branch, dispatch.version, rid=dispatch.action.rid)])
        except EngineUnavailable:
            with suppress(httpx.HTTPError):
                await self.http.post("/abort_request", json={"rid": dispatch.action.rid}, timeout=2)

    async def finish_native(self, dispatch: Dispatch) -> None:
        action = dispatch.action
        try:
            reply = await dispatch.future
            details = reply.get("details", {})
            version = dispatch.version
            if action.kind == "prepare" and reply["status"] == "ready" and reply.get("handle"):
                version.handle = reply["handle"]
                version.prepare_lease = True
                if action.status in {"canceled", "error"}:
                    self.drop_candidate_lease(dispatch.branch, version)
            if reply["status"] not in _DONE:
                self.directory.settle_engine(action, "canceled" if reply["status"] == "canceled" else "error", reply.get("error"))
                dispatch.finished.set_result(reply)
                return
            actual = details.get("token_ids")
            if actual != list(dispatch.expected) or action.status in {"canceled", "error"}:
                self.directory.settle_engine(action, "error", "scheduler did not confirm the bound token sequence")
                dispatch.finished.set_result({**reply, "status": "error", "error": action.error or "request not confirmed"})
                return
            version.tokens = dispatch.expected
            version.handle = reply.get("handle")
            version.residency = details.get("residency", "absent")
            version.matched = reply["matched_tokens"]
            version.full_hit = details.get("full_kv_hit_length", 0)
            if action.kind == "prepare":
                candidate = dispatch.branch.candidate
                if (candidate is not None and candidate.action_id == action.id and
                        candidate.status == "preparing" and version.prepare_lease):
                    candidate.status = "ready"
            self.directory.settle_engine(action, "completed")
            dispatch.finished.set_result(reply)
        finally:
            self.retire(dispatch)

    async def complete(self, dispatch: Dispatch) -> EngineReply:
        remaining = max(0.0, (dispatch.action.deadline_ns - time.monotonic_ns()) / 1e9)
        try:
            return await asyncio.wait_for(asyncio.shield(dispatch.finished), remaining)
        except TimeoutError:
            self.cancel(dispatch.action, "missing native terminal ACK at action deadline")
            raise WireError(504, "engine_ack_timeout", "scheduler did not confirm terminal state") from None

    def drop_candidate_lease(self, branch: Branch, version: Version) -> None:
        if version.prepare_lease:
            version.prepare_lease = False
            self.spawn(self.release(branch, version, epoch=self.engine.epoch))

    async def release(self, branch: Branch, version: Version, *, handle: str | None = None,
                      epoch: str | None = None, residency: str = "hbm") -> None:
        handle = handle or version.handle
        if handle and self.engine.epoch is not None and epoch in (None, self.engine.epoch):
            with suppress(EngineUnavailable):
                command = self.command("release", branch, version, payload={"residency": residency})
                command["handle"] = handle
                await self.engine.rpc([command])

    async def pin_fork(self, branch: Branch) -> None:
        version, epoch = branch.inherited, self.engine.epoch
        if version is None or not version.handle or epoch is None:
            return
        handle, residency = version.handle, version.residency
        if residency not in ("hbm", "host"):
            return
        with suppress(EngineUnavailable):
            reply, = await self.engine.rpc([self.command("lease", branch, version, payload={"residency": residency})])
            if reply["status"] == "ok":
                if branch.lifecycle == "expired":
                    await self.release(branch, version, handle=handle, epoch=epoch, residency=residency)
                    return
                self.fork_leases[branch.key] = (version, handle, epoch, residency)

    async def move(self, move: Movement) -> None:
        c, branch, version = move.candidate, move.branch, move.version
        aid = f"{c.action_id}/{uuid.uuid4().hex}"
        future = self.engine.watch(aid)
        start = time.monotonic_ns()
        try:
            if not version.handle or self.checkpoint(branch) is not version or start > c.expires_mono_ns:
                return
            if move.op == "retain":
                if branch.lifecycle != "active" or branch.owner is not None or self.engine.epoch is None:
                    return
                handle, epoch = version.handle, self.engine.epoch
                reply, = await self.engine.rpc([self.command("lease", branch, version, aid=aid)])
                self.receipt(reply)
                if reply["status"] == "ok":
                    self.retained[branch.key] = (version, handle, epoch)
                return
            payload = None
            if move.op == "reclaim":
                if (branch.owner is not None
                    or (branch.lifecycle != "expired"
                        and not (branch.lifecycle == "suspended" and self.policy.mode == "ttl"))
                    or any(b.lifecycle != "expired" and b is not branch
                           and (b.inherited is version or self.checkpoint(b) is version)
                           for b in self.directory.branches.values())):
                    return
                payload = {"residency": version.residency,
                           "discard_unbacked": branch.lifecycle == "expired"}
            reply, = await self.engine.rpc([self.command(move.op, branch, version, aid=aid, payload=payload)])
            if reply["status"] == "accepted":
                reply = await future  # Do not open the background gate before native ACK/epoch loss.
            else:
                self.receipt(reply)
            if reply["status"] in _DONE:
                if c.wait_type and reply.get("details", {}).get("transfer_metrics_complete"):
                    transfer = reply["transfer_us"] / 1000
                    queue = max(0.0, (time.monotonic_ns() - start) / 1e6 - transfer)
                    self.policy.observe(c.wait_type, transfer, queue)
                version.residency = reply.get("details", {}).get("residency", version.residency)
                self.move_cooldown[version.id] = time.monotonic_ns()
        except EngineUnavailable:
            pass
        finally:
            self.move_cooldown[c.action_id] = time.monotonic_ns()
            self.engine.unwatch(aid)
            self.movements.pop(c.action_id, None)
            move.done.set()

    async def pretokenize(self, action: Action) -> None:
        try:
            body = _chat_body(action.payload["chat"])
            ids = await self.tokenize(body)
            _context(ids, 0)
            branch = self.directory.branches[action.key]
            candidate = branch.candidate
            if action.status == "ready" and candidate is not None and candidate.action_id == action.id:
                if len(ids) < 2:
                    self.cancel(action, "no reusable tokens before the final generation token")
                    return
                candidate.target_tokens = ids
                # Native autoregressive admission caps cache matching at
                # input_len-1. Mamba's checkpoint must exist at that boundary.
                self.preparing[action.id] = ids[:-1]
        except (WireError, httpx.HTTPError) as exc:
            self.cancel(action, str(exc))

    async def prepare(self, action: Action) -> None:
        branch = self.directory.branches[action.key]
        candidate = branch.candidate
        ids = self.preparing.pop(action.id, None)
        if candidate is None or candidate.action_id != action.id or ids is None:
            self.cancel(action, "candidate invalidated before prefill")
            return
        try:
            self.directory.start(action, branch)
            dispatch = await self.bind(action, candidate.version, ids, 0)
            result = await self.http.post("/generate", json={"rid": action.rid, "input_ids": list(ids),
                "cache_salt": dispatch.salt, "sampling_params": {"max_new_tokens": 0, "temperature": 0}, "stream": False})
            result.raise_for_status()
            reply = await self.complete(dispatch)
            if reply["status"] != "ready" or candidate.status != "ready":
                self.directory._discard(branch, candidate)
        except (WireError, httpx.HTTPError) as exc:
            self.cancel(action, str(exc))
        finally:
            if candidate.status == "canceled":
                self.drop_candidate_lease(branch, candidate.version)

    def cutover(self, branch: Branch, ids: tuple[int, ...]) -> Version | None:
        candidate = branch.candidate_cutover
        if candidate is None:
            return None
        base, current = candidate.version, branch.current
        target = candidate.target_tokens
        if target and ids[:len(target)] == target and base.prepare_lease:
            current.history = base.history
            branch.candidate_cutover = None
            return base
        self.directory._discard(branch, candidate)
        return None

    async def message_ranges(self, body: dict[str, Any], version: Version, ids: tuple[int, ...]) -> None:
        """Record each message's token span in ``ids`` for later message-boundary drops.

        The chat template renders messages additively for this carrier; the spans
        are optimistic and the engine's ``repos`` validation is the authoritative
        check — on any mismatch the drop falls back to a cold rebuild. Ranges are
        only maintained once a session grows large enough for a drop to be
        plausible, then incrementally for the messages each turn adds.
        """
        messages = body.get("messages") or []
        parent = version.parent
        base = parent.ranges if parent is not None and parent.ranges is not None else ()
        if len(messages) != len(version.message_ids) or len(base) > len(messages):
            return
        if not base and len(messages) < 4:
            # Below four messages no middle exists for a drop to remove.
            return
        kwargs = dict(body.get("chat_template_kwargs") or {})
        ranges = list(base)
        prefix_end = sum(end - start for start, end in base)
        dummy = {"role": "user", "content": "Continue."}
        dummy_tokens = None
        # Tokenize growing prefixes of the real message list; the template is
        # prefix-stable for this carrier, so message j's span is the delta
        # between adjacent prefix lengths. The engine repos validation remains
        # the authoritative check of every boundary. A leading system message
        # is rendered before a trailing dummy user because the engine rejects
        # lists without a user query.
        for index in range(len(base), len(messages)):
            try:
                if index == 0 and messages[0].get("role") in ("system", "developer"):
                    if dummy_tokens is None:
                        dummy_tokens = await self.tokenize({"messages": [dummy], "chat_template_kwargs": kwargs,
                                                            "add_generation_prompt": False})
                    rendered = await self.tokenize({"messages": [messages[0], dummy],
                                                    "chat_template_kwargs": kwargs,
                                                    "add_generation_prompt": False})
                    end = len(rendered) - len(dummy_tokens)
                else:
                    rendered = await self.tokenize({"messages": messages[:index + 1], "tools": body.get("tools"),
                                                    "tool_choice": body.get("tool_choice"),
                                                    "chat_template_kwargs": kwargs,
                                                    "add_generation_prompt": False})
                    end = len(rendered)
            except WireError:
                return
            ranges.append((prefix_end, end))
            prefix_end = end
        version.ranges = tuple(ranges)

    async def materialize_source(self, branch: Branch, source: Version) -> bool:
        """Checkpoint the committed view so a repos op has a source handle.

        Completed chat requests do not enter the hybrid tree under
        ``mamba_radix_cache_strategy=no_buffer``; the prepare path materialises
        the chain and returns a handle. The prepare hits the previous chain, so
        only the un-checkpointed delta is prefilled.
        """
        aid = uuid.uuid4().hex
        rid = uuid.uuid4().hex
        salt = self.directory.salt(branch, source, source.tokens or ())
        try:
            reply, = await self.engine.rpc([{
                "op": "bind", "action_id": aid, "rid": rid, "run_id": branch.run_id,
                "version_id": source.id, "handle": None,
                "payload": {"expected_ids": list(source.tokens or ()), "cache_salt": salt,
                            "kind": "prepare", "session_id": branch.key[0], "branch_id": branch.key[1],
                            "max_new_tokens": 0, "hold": False}}])
        except EngineUnavailable:
            return False
        if reply.get("status") != "ok":
            return False
        future = self.engine.watch(aid)
        native = self.http.build_request("POST", "/generate", json={
            "input_ids": list(source.tokens or ()), "rid": rid, "cache_salt": salt,
            "sampling_params": {"max_new_tokens": 0, "temperature": 0, "ignore_eos": True}})
        try:
            response = await self.http.send(native)
            response.raise_for_status()
            receipt = await future
        except (httpx.HTTPError, EngineUnavailable):
            self.engine.unwatch(aid)
            return False
        finally:
            self.engine.unwatch(aid)
        if receipt.get("status") != "ready" or not receipt.get("handle"):
            return False
        source.handle = receipt["handle"]
        source.residency = (receipt.get("details") or {}).get("residency", "hbm")
        source.full_hit = (receipt.get("details") or {}).get("full_kv_hit_length", 0)
        LOG.info("repos source prepared %s", json.dumps({
            "run_id": branch.run_id, "session_id": branch.key[0], "branch_id": branch.key[1],
            "version_id": source.id, "tokens": len(source.tokens or ()), "full_hit": source.full_hit}))
        return True

    async def materialize_repos(self, branch: Branch) -> bool:
        """Execute the harness's message-level drop via the engine ``repos`` op.

        The kept view's token sequence is the source minus the dropped messages'
        recorded spans, so no re-tokenization happens here; the engine validates
        the spans against the source chain before rotating. Every failure mode
        (no source tokens/ranges, invalid subsequence, engine rejection) leaves
        the version cold — the next request rebuilds it honestly.
        """
        version = branch.current
        source = version.parent
        if (source is None or source.tokens is None
                or not source.ranges or len(source.ranges) != len(source.message_ids)
                or not self.engine.capabilities.get("repos")):
            LOG.info("repos cold fallback %s", json.dumps({
                "run_id": branch.run_id, "session_id": branch.key[0], "branch_id": branch.key[1],
                "version_id": version.id, "reason": "source not fully materialised"}))
            return False
        if (source.handle is None or source.residency != "hbm") and not await self.materialize_source(branch, source):
            LOG.info("repos cold fallback %s", json.dumps({
                "run_id": branch.run_id, "session_id": branch.key[0], "branch_id": branch.key[1],
                "version_id": version.id, "reason": "source prepare failed"}))
            return False
        kept_ids = version.message_ids
        by_id = dict(zip(source.message_ids, source.ranges))
        kept_source = [mid for mid in source.message_ids if mid in set(kept_ids)]
        appended = [mid for mid in kept_ids if mid not in by_id]
        if tuple(kept_ids) != tuple(kept_source + appended):
            LOG.info("repos cold fallback %s", json.dumps({
                "run_id": branch.run_id, "session_id": branch.key[0], "branch_id": branch.key[1],
                "version_id": version.id, "reason": "kept messages are not a clean subsequence plus appended tail"}))
            return False
        spans: list[tuple[int, int]] = []
        kept_ranges: list[tuple[int, int]] = []
        offset = 0
        for mid in source.message_ids:
            start, end = by_id[mid]
            if mid in set(kept_ids):
                kept_ranges.append((offset, offset + end - start))
                offset += end - start
            else:
                spans.append((start, end))
        if not spans:
            return False
        compact = tuple(token for index, token in enumerate(source.tokens)
                        if not any(start <= index < end for start, end in spans))
        if len(compact) == len(source.tokens):
            LOG.info("repos cold fallback %s", json.dumps({
                "run_id": branch.run_id, "session_id": branch.key[0], "branch_id": branch.key[1],
                "version_id": version.id, "reason": "message spans covered no tokens"}))
            return False
        command = {"op": "repos", "action_id": uuid.uuid4().hex, "rid": uuid.uuid4().hex,
                   "run_id": branch.run_id, "version_id": source.id, "handle": source.handle,
                   "payload": {"expected_ids": list(compact), "cache_salt": self.directory.salt(branch, version, compact),
                               "dropped": [list(span) for span in spans]}}
        try:
            reply, = await self.engine.rpc([command])
        except EngineUnavailable:
            return False
        if reply.get("status") != "completed":
            LOG.info("repos cold fallback %s", json.dumps({
                "run_id": branch.run_id, "session_id": branch.key[0], "branch_id": branch.key[1],
                "version_id": version.id, "reason": reply.get("error") or "repos rejected"}))
            return False
        version.tokens = compact
        version.ranges = tuple(kept_ranges)
        version.residency = "hbm"
        version.matched = reply.get("matched") or 0
        version.full_hit = reply.get("matched") or 0
        version.handle = reply.get("handle")
        LOG.info("repos materialised %s", json.dumps({
            "run_id": branch.run_id, "session_id": branch.key[0], "branch_id": branch.key[1],
            "version_id": version.id, "kept_tokens": len(compact), "source_tokens": len(source.tokens),
            "dropped_spans": spans, "rotated_bytes": (reply.get("details") or {}).get("rotated_bytes")}))
        return True


def _chat_body(value: Any) -> dict[str, Any]:
    body = object_value(value, "chat request").copy()
    if any(key in body for key in ("rid", "cache_salt", "extra_key", "session_params", "kv_hints")):
        raise invalid("client-supplied engine routing or cache policy is forbidden")
    if body.get("model", MODEL) != MODEL or not isinstance(body.get("messages"), list):
        raise invalid("expected the pinned model and messages array")
    if type(body.get("n", 1)) is not int or body.get("n", 1) != 1 or type(body.get("stream", False)) is not bool:
        raise invalid("only one completion and boolean stream are supported")
    kwargs = object_value(body.get("chat_template_kwargs", {}), "chat_template_kwargs").copy()
    kwargs["enable_thinking"] = False
    body.update(model=MODEL, temperature=0, chat_template_kwargs=kwargs)
    return body


def _wire_failure(action: Action) -> WireError:
    return WireError(409 if action.status == "canceled" else 503, "action_not_granted", action.error or action.status)


def _response_headers(response: httpx.Response) -> dict[str, str]:
    return {k: v for k, v in response.headers.items() if k.lower() in {"content-type", "cache-control", "x-request-id"}}


def _terminal_chat_frame(frame: bytes) -> bool:
    if not frame.startswith(b"data:"):
        return False
    data = frame[5:].strip()
    if data == b"[DONE]":
        return True
    payload = json.loads(data)
    return "error" in payload or any(choice.get("finish_reason") is not None
                                     for choice in payload.get("choices", ()))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(base_url=settings.backend, timeout=httpx.Timeout(900, connect=2),
                                     trust_env=False, limits=httpx.Limits(max_connections=32)) as http:
            service = Coordinator(settings, http)
            app.state.service = service
            await service.start()
            try:
                yield
            finally:
                await service.close()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authorized(request: Request) -> Coordinator:
        auth = request.headers.get("authorization", "")
        if not hmac.compare_digest(auth.encode(), f"Bearer {settings.bearer}".encode()):
            raise WireError(401, "unauthorized", "invalid local bearer")
        return request.app.state.service

    @app.exception_handler(WireError)
    async def handle_wire(_request: Request, exc: WireError):
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(httpx.HTTPError)
    async def handle_transport(_request: Request, _exc: httpx.HTTPError):
        return JSONResponse(EngineUnavailable("native HTTP transport failed").body(), status_code=503)

    @app.get("/health")
    async def health(request: Request):
        service = authorized(request)
        if service.engine.epoch is None:
            raise EngineUnavailable("no active scheduler epoch")
        return {"epoch": service.engine.epoch, "capabilities": service.engine.capabilities}

    @app.get("/v1/models")
    async def models(request: Request):
        result = await authorized(request).http.get("/v1/models")
        return Response(result.content, status_code=result.status_code, headers=_response_headers(result))

    @app.get("/agentic/v1/events/{run_id}/{event_id}")
    async def event_status(run_id: str, event_id: str, request: Request):
        service = authorized(request)
        record = service.directory.events.get((run_id, event_id))
        if record is None:
            raise unknown("event not recorded")
        return record.ack()

    @app.post("/agentic/v1/events")
    async def events(request: Request):
        service = authorized(request)
        event = Event.parse(await _body(request))
        duplicate = (event.run_id, event.event_id) in service.directory.events
        record = service.directory.accept(event)
        if not duplicate:
            payload = event.payload
            aid = payload.get("action_id")
            action = service.directory.actions.get(aid) if isinstance(aid, str) else None
            LOG.info("event %s", json.dumps({
                "mono_ns": time.monotonic_ns(), "run_id": event.run_id, "session_id": event.session_id,
                "branch_id": event.branch_id, "seq": event.seq, "event_id": event.event_id,
                "version_id": record.version_id, "kind": event.kind, "status": record.ack()["status"],
                "payload": {k: payload[k] for k in ("action_id", "rid", "action_kind", "turn_id", "call_id",
                             "ordinal", "status", "cache_hit", "wait_ms", "children") if k in payload},
                "rid": action.rid if action else None,
            }, separators=(",", ":")))
        if not duplicate and event.kind == EventKind.FINISH and event.payload.get("cache_hit") is True:
            branch = service.directory.branch((event.session_id, event.branch_id))
            if branch.candidate is not None:
                prepare_action = service.directory.actions[branch.candidate.action_id]
                if prepare_action.status in ("ready", "granted"):
                    service.cancel(prepare_action, "confirmed tool cache hit ended the wait")
        service.wake.set()
        if record.action is not None:
            action = record.action
            if action.kind == "prepare":
                # An optional proposal is recorded, not a permit dependency of
                # the foreground. The harness polls the same event for READY.
                if not duplicate:
                    service.spawn(service.pretokenize(action))
            else:
                await _connected(request, service.grant(action), lambda: service.cancel(action, "ready caller disconnected"))
        if not duplicate and event.kind == EventKind.REPOS:
            branch = service.directory.branch((event.session_id, event.branch_id))
            if await service.materialize_repos(branch):
                record.status = "ready"
        ack = record.ack()
        if event.kind == EventKind.FORK:
            key = (event.payload.get("child_session_id", event.session_id), event.payload["child_branch_id"])
            if not duplicate:
                job = service.spawn(service.pin_fork(service.directory.branches[key]))
                service.fork_pins[key] = job
                job.add_done_callback(lambda _job: service.fork_pins.pop(key, None))
            if pin := service.fork_pins.get(key):
                await asyncio.shield(pin)
            ack = {**ack, "child_version_id": service.directory.fork_versions[key]}
        return ack

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        service = authorized(request)
        body = _chat_body(await _body(request))
        headers = request.headers
        key = (identifier(headers.get(SESSION_HEADER), SESSION_HEADER), identifier(headers.get(BRANCH_HEADER), BRANCH_HEADER))
        branch = service.directory.branch(key)
        version_id = identifier(headers.get(VERSION_HEADER), VERSION_HEADER)
        rid = identifier(headers.get(REQUEST_HEADER), REQUEST_HEADER)
        action = service.directory.rids.get(rid)
        if action is None or action.key != key or action.version_id != version_id or action.kind not in ("model", "summary"):
            raise unknown("no matching live request handle")
        if action.status != "granted":
            raise conflict("rid does not hold an unused permit")
        version = branch.current
        budget = _budget(body.pop("max_tokens", MAX_GENERATE) if "max_completion_tokens" not in body else body.pop("max_completion_tokens"), MAX_GENERATE)
        body.pop("max_tokens", None)
        body["max_completion_tokens"] = budget
        service.directory.start(action, branch)
        dispatch: Dispatch | None = None
        cutover = None
        try:
            ids = await _connected(request, service.tokenize(body), lambda: service.cancel(action, "client disconnected"))
            _context(ids, budget)
            cutover = service.cutover(branch, ids)
            await service.message_ranges(body, version, ids)
            dispatch = await service.bind(action, version, ids, budget)
            native = service.http.build_request("POST", "/v1/chat/completions", json={**body, "rid": rid, "cache_salt": dispatch.salt})
            upstream = await _connected(request, service.http.send(native, stream=body.get("stream", False)),
                                        lambda: service.cancel(action, "client disconnected"))
        except BaseException as exc:
            if dispatch is not None or action.id in service.inflight:
                service.cancel(action, "chat ingress failed")
            else:
                service.directory.settle_engine(action, "error", str(exc))
            raise
        finally:
            if cutover is not None:
                service.drop_candidate_lease(branch, cutover)
        if upstream.is_error:
            try:
                content = await upstream.aread()
            finally:
                await upstream.aclose()
                service.cancel(action, "native chat rejected request")
            return Response(content, status_code=upstream.status_code, headers=_response_headers(upstream))
        if not body.get("stream", False):
            reply = await _connected(request, service.complete(dispatch), lambda: service.cancel(action, "client disconnected"))
            if reply["status"] not in _DONE:
                raise EngineUnavailable(reply.get("error") or "native generation failed")
            return Response(upstream.content, status_code=upstream.status_code, headers=_response_headers(upstream))

        async def proxy():
            pending = bytearray()
            confirmed = False
            try:
                async for chunk in upstream.aiter_bytes():
                    pending.extend(chunk)
                    while (end := pending.find(b"\n\n")) >= 0:
                        frame = bytes(pending[:end + 2])
                        del pending[:end + 2]
                        if not confirmed and _terminal_chat_frame(frame):
                            reply = await service.complete(dispatch)
                            if reply["status"] not in _DONE:
                                raise EngineUnavailable(reply.get("error") or "native generation failed")
                            confirmed = True
                        yield frame
                if pending or not confirmed:
                    raise EngineUnavailable("native chat did not return a complete SSE stream")
            except (WireError, httpx.HTTPError, ValueError) as exc:
                message = exc.body() if isinstance(exc, WireError) else EngineUnavailable("native SSE transport failed").body()
                yield b"data: " + json.dumps(message).encode() + b"\n\n"
            finally:
                await upstream.aclose()
                if action.id in service.inflight:
                    service.cancel(action, "chat stream did not finish")

        return StreamingResponse(proxy(), status_code=upstream.status_code, headers=_response_headers(upstream))

    @app.post("/agentic/v1/batch")
    async def batch(request: Request):
        service = authorized(request)
        raw = object_value(await _body(request), "batch request")
        items = raw.get("items")
        if set(raw) != {"items"} or not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH:
            raise invalid(f"batch must contain 1..{MAX_BATCH} items")
        return await run_batch(request, service, items)

    return app


def _batch_params(raw: Any) -> dict[str, Any]:
    params = object_value(raw, "sampling_params")
    allowed = {"max_new_tokens", "temperature", "ignore_eos", "n"}
    n, temperature = params.get("n", 1), params.get("temperature", 0)
    if params.keys() - allowed or type(n) is not int or n != 1 or type(temperature) not in (int, float) or temperature != 0:
        raise invalid("replay must use unmodified greedy single-sample token generation")
    if params.get("ignore_eos", True) is not True:
        raise invalid("replay requires ignore_eos")
    return {"temperature": 0, "max_new_tokens": _budget(params.get("max_new_tokens"), MAX_REPLAY_GENERATE),
            "ignore_eos": True}


async def run_batch(request: Request, service: Coordinator, items: list[Any]) -> dict[str, Any]:
    result: list[dict[str, Any] | None] = [None] * len(items)
    staged: list[tuple[int, Action, Version, tuple[int, ...], dict[str, Any]]] = []
    branch_keys: set[tuple[str, str]] = set()
    rids: set[str] = set()
    for i, raw in enumerate(items):
        rid = raw.get("rid") if isinstance(raw, dict) else None
        try:
            item = object_value(raw, "batch item")
            if set(item) != {"session_id", "branch_id", "version_id", "rid", "input_ids", "sampling_params"}:
                raise invalid("invalid batch item fields")
            rid = identifier(rid, "rid")
            key = (identifier(item["session_id"], "session_id"), identifier(item["branch_id"], "branch_id"))
            if key in branch_keys or rid in rids:
                raise conflict("duplicate branch or rid in batch")
            branch_keys.add(key)
            rids.add(rid)
            branch = service.directory.branch(key)
            version = branch.current
            if item["version_id"] != version.id or branch.owner is not None:
                raise conflict("batch version is stale or branch is busy")
            ids, params = token_ids(item["input_ids"]), _batch_params(item["sampling_params"])
            _context(ids, params["max_new_tokens"])
            action = service.directory.rids.get(rid)
            if action is None:
                action = service.directory._ready(branch, {"action_id": uuid.uuid4().hex, "rid": rid,
                    "action_kind": "batch", "turn_id": rid, "call_id": rid, "ordinal": i})
            if action.key != key or action.version_id != version.id or action.kind != "batch" or action.status not in ("ready", "granted"):
                raise conflict("rid has no compatible unused batch permit")
            staged.append((i, action, version, ids, params))
        except WireError as exc:
            result[i] = {"rid": rid, "status": "error", **exc.body()}
    service.wake.set()

    def cancel_all() -> None:
        for _, action, *_ in staged:
            service.cancel(action, "batch client disconnected")

    async def execute() -> None:
        waiters = {asyncio.create_task(service.grant(entry[1])): entry for entry in staged}
        try:
            while waiters:
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                granted = []
                for task, entry in tuple(waiters.items()):
                    if entry[1].status != "ready":
                        granted.append(entry)
                        waiters.pop(task)
                        task.cancel()
                await send_group(granted)
        finally:
            for task in waiters:
                task.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)

    async def send_group(group) -> None:
        dispatches: list[tuple[int, Dispatch, dict[str, Any]]] = []
        for i, action, version, ids, params in group:
            try:
                if action.status != "granted":
                    raise _wire_failure(action)
                service.directory.start(action, service.directory.branches[action.key])
                dispatches.append((i, await service.bind(action, version, ids, params["max_new_tokens"]), params))
            except WireError as exc:
                if action.id not in service.inflight:
                    service.directory.settle_engine(action, "error", str(exc))
                result[i] = {"rid": action.rid, "status": "canceled" if action.status == "canceled" else "error", **exc.body()}
        if not dispatches:
            return
        # Native nonstream generation rejects the whole batch if one item
        # returns an HTTP-class abort; native streaming retains that item.
        native = {"rid": [d.action.rid for _, d, _ in dispatches], "input_ids": [list(d.expected) for _, d, _ in dispatches],
                  "cache_salt": [d.salt for _, d, _ in dispatches], "sampling_params": [p for _, _, p in dispatches],
                  "stream": True}
        by_rid = {}
        error = None
        try:
            async with service.http.stream("POST", "/generate", json=native) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    item = json.loads(data)
                    meta = item.get("meta_info", {})
                    if meta.get("finish_reason"):
                        by_rid[meta["id"]] = item
                    if "error" in item:
                        error = item["error"]
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            error = {"message": str(exc)}

        async def collect(i: int, d: Dispatch) -> None:
            rid = d.action.rid
            item = by_rid.get(rid)
            if item is None and error is not None:
                service.cancel(d.action, "native batch transport/request error")
                result[i] = {"rid": rid, "status": "error", "error": error}
                return
            try:
                reply = await service.complete(d)
            except WireError as exc:
                result[i] = {"rid": rid, "status": "error", **exc.body()}
                return
            reason = (item or {}).get("meta_info", {}).get("finish_reason") or {}
            if d.action.status == "canceled" or reply["status"] == "canceled":
                result[i] = {"rid": rid, "status": "canceled", "error": reply.get("error") or reason}
            elif reply["status"] not in _DONE or item is None or reason.get("type") == "abort":
                result[i] = {"rid": rid, "status": "error", "error": reply.get("error") or reason or "missing native batch output"}
            else:
                result[i] = {"rid": rid, "status": "completed", "result": item}

        await asyncio.gather(*(collect(i, d) for i, d, _ in dispatches))

    try:
        await _connected(request, execute(), cancel_all)
    except BaseException:
        cancel_all()
        raise
    return {"items": result}


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=30081)


if __name__ == "__main__":
    main()
