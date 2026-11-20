"""Shared wire contract; no runtime or engine imports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, TypedDict, NotRequired

MAX_LINE = 1 << 20  # Includes the LF delimiter.
MAX_BATCH = 8
MAX_CONTEXT = 32768
MAX_GENERATE = 512
MAX_REPLAY_GENERATE = 128
PROTOCOL_VERSION = 1
SESSION_HEADER = "X-Agentic-Session"
BRANCH_HEADER = "X-Agentic-Branch"
VERSION_HEADER = "X-Agentic-Version"
REQUEST_HEADER = "X-Agentic-Request"
PRIVATE_HEADERS = (SESSION_HEADER, BRANCH_HEADER, VERSION_HEADER, REQUEST_HEADER)
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@/-]{0,255}\Z", re.ASCII)


class EventKind(StrEnum):
    APPEND = "append"
    REPLACE = "replace"
    REMOVE = "remove"
    REPOS = "repos"
    FORK = "fork"
    END = "end"
    SUSPEND = "suspend"
    RESUME = "resume"
    CANCEL = "cancel"
    READY = "ready"
    START = "start"
    FINISH = "finish"
    COMMIT = "commit"
    JOIN = "join"


class EngineOp(StrEnum):
    BIND = "bind"
    LEASE = "lease"
    RELEASE = "release"
    RESTORE = "restore"
    OFFLOAD = "offload"
    RECLAIM = "reclaim"
    REPOS = "repos"
    CANCEL = "cancel"
    SNAPSHOT = "snapshot"


class EngineStatus(StrEnum):
    OK = "ok"
    ACCEPTED = "accepted"
    READY = "ready"
    PARTIAL = "partial"
    COMPLETED = "completed"
    CANCELED = "canceled"
    STALE = "stale"
    NOT_FOUND = "not_found"
    UNAVAILABLE = "unavailable"
    ERROR = "error"


class Permit(TypedDict):
    action_id: str
    version_id: str
    expires_mono_ns: int


class EventAck(TypedDict):
    event_id: str
    version_id: str
    status: str
    permit: Permit | None


class EngineCommand(TypedDict):
    op: str
    action_id: str
    run_id: str
    rid: str
    version_id: str
    handle: str | None
    # bind: expected_ids is the exact tokenization of this request, not the
    # parent. kind=chat|batch|prepare. hold pins a completed prepare until release.
    payload: NotRequired[dict[str, Any]]


class EngineRequest(TypedDict):
    epoch: str | None
    ops: list[EngineCommand]


class EngineReply(TypedDict):
    epoch: str
    seq: int
    action_id: str
    status: str
    version_id: str
    matched_tokens: int
    full_kv_slots: int
    recurrent_slots: int
    unique_hbm_bytes: int
    unique_host_bytes: int
    reserved_bytes: int
    fg_queue: int
    bg_queue: int
    transfer_bytes: int
    transfer_us: int
    error: str | None
    run_id: NotRequired[str]
    rid: NotRequired[str]
    handle: NotRequired[str | None]
    # snapshot: protocol, capabilities, pools, and recent completion receipts.
    # completion: token_ids (actual input), full_kv_hit_length, residency.
    details: NotRequired[dict[str, Any]]


class EngineResponse(TypedDict):
    epoch: str
    # Explicit operations are ordered; empty ops returns one snapshot reply.
    replies: list[EngineReply]


class WireError(ValueError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code

    def body(self) -> dict[str, Any]:
        return {"error": {"code": self.code, "message": str(self)}}


def invalid(message: str) -> WireError:
    return WireError(422, "invalid_request", message)


def conflict(message: str) -> WireError:
    return WireError(409, "conflict", message)


def unknown(message: str) -> WireError:
    return WireError(404, "unknown_handle", message)


def identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise invalid(f"invalid {name}")
    return value


def object_value(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise invalid(f"{name} must be an object")
    return value


def token_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) > MAX_CONTEXT or any(
        type(token) is not int or not 0 <= token < 2**31 for token in value
    ):
        raise invalid("invalid input_ids")
    return tuple(value)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON constant {value}")


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    merged = dict(pairs)
    if len(merged) != len(pairs):
        raise ValueError("duplicate JSON key")
    return merged


def encode_line(value: Any) -> bytes:
    data = canonical_json(value) + b"\n"
    if len(data) > MAX_LINE:
        raise invalid("engine JSON line exceeds 1 MiB")
    return data


def decode_line(data: bytes) -> dict[str, Any]:
    if not data.endswith(b"\n") or len(data) > MAX_LINE or b"\n" in data[:-1]:
        raise invalid("engine message must be a single LF-terminated line of at most 1 MiB")
    try:
        return object_value(json.loads(data.decode("utf-8"), parse_constant=_reject_constant,
                                       object_pairs_hook=_unique_keys), "engine message")
    except (ValueError, RecursionError) as exc:
        raise invalid("invalid engine JSON") from exc


def parse_engine_request(value: Any) -> EngineRequest:
    raw = object_value(value, "engine request")
    if set(raw) != {"epoch", "ops"} or not isinstance(raw["ops"], list) or len(raw["ops"]) > 64:
        raise invalid("invalid engine envelope")
    if raw["epoch"] is None:
        if raw["ops"]:
            raise invalid("null epoch is only allowed for initial snapshot")
    else:
        identifier(raw["epoch"], "epoch")
    for command in raw["ops"]:
        op = object_value(command, "operation")
        if not {"op", "action_id", "run_id", "rid", "version_id", "handle"} <= op.keys():
            raise invalid("incomplete engine operation")
        try:
            EngineOp(op["op"])
        except (ValueError, TypeError) as exc:
            raise invalid("unknown engine operation") from exc
        for key in ("action_id", "rid", "version_id", "run_id"):
            identifier(op[key], key)
        if op["handle"] is not None:
            identifier(op["handle"], "handle")
        if "payload" in op:
            object_value(op["payload"], "operation payload")
    return raw  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class Event:
    run_id: str
    session_id: str
    branch_id: str
    seq: int
    event_id: str
    expected_version: str | None
    kind: EventKind
    payload: dict[str, Any]
    identity: bytes

    @classmethod
    def parse(cls, value: Any) -> Event:
        raw = object_value(value, "event")
        if set(raw) != {"run_id", "session_id", "branch_id", "seq", "event_id",
                        "expected_version", "kind", "payload"}:
            raise invalid("invalid event fields")
        if type(raw["seq"]) is not int or not 1 <= raw["seq"] < 2**53:
            raise invalid("invalid seq")
        try:
            kind = EventKind(raw["kind"])
            identity = canonical_json(raw)
        except (TypeError, ValueError, RecursionError) as exc:
            raise invalid("invalid event kind or payload") from exc
        return cls(identifier(raw["run_id"], "run_id"), identifier(raw["session_id"], "session_id"),
                   identifier(raw["branch_id"], "branch_id"), raw["seq"],
                   identifier(raw["event_id"], "event_id"),
                   None if raw["expected_version"] is None else identifier(raw["expected_version"], "expected_version"),
                   kind, object_value(raw["payload"], "event payload"), identity)
