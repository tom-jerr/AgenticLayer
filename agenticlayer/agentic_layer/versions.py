"""Logical state owned by ingress. Native allocation never lives in this module."""
from __future__ import annotations

import asyncio
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from .protocol import (
    Event, EventAck, EventKind, Permit, canonical_json, conflict, identifier, invalid, unknown,
)


TERMINAL = frozenset({"completed", "canceled", "error", "committed"})
PERMIT_NS = 2_000_000_000
DEADLINE_NS = 900_000_000_000
BranchKey = tuple[str, str]  # session_id, branch_id; bound to one run


def digest(*values: Any) -> str:
    return hashlib.sha256(canonical_json(values)).hexdigest()


def _message_ids(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise invalid("message_ids must be an ordered array")
    if any(not isinstance(item, str) or not item or len(item) > 512 for item in value):
        raise invalid("invalid message identity")
    if len(set(value)) != len(value):
        raise invalid("duplicate message identity")
    return tuple(value)


@dataclass(slots=True)
class Version:
    id: str
    message_ids: tuple[str, ...]
    surface_hash: str | None
    history: str
    parent: Version | None = None
    fork_prefix: tuple[int, ...] | None = None
    tokens: tuple[int, ...] | None = None
    handle: str | None = None
    residency: str = "absent"
    matched: int = 0
    full_hit: int = 0
    prepare_lease: bool = False


@dataclass(slots=True)
class Action:
    id: str
    rid: str
    run_id: str
    key: BranchKey
    version_id: str
    kind: str
    turn: str
    call: str
    ordinal: int
    ready_ns: int
    deadline_ns: int
    dependencies: tuple[str, ...]
    payload: dict[str, Any]
    status: str = "ready"
    permit: Permit | None = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    draining: bool = False
    committed: bool = False
    error: str | None = None

    def set_status(self, status: str, error: str | None = None) -> None:
        self.status, self.error = status, error
        self.changed.set()


@dataclass(slots=True)
class CandidateVersion:
    action_id: str
    base: Version
    boundary: tuple[str, ...]
    expected_target: tuple[str, ...] | None
    version: Version
    status: str = "preparing"
    target_tokens: tuple[int, ...] | None = None


@dataclass(slots=True)
class Branch:
    run_id: str
    key: BranchKey
    domain: str
    current: Version
    seq: int = 0
    lifecycle: str = "active"
    last_ended_turn: Version | None = None
    candidate: CandidateVersion | None = None
    action_ids: set[str] = field(default_factory=set)
    owner: str | None = None
    last_activity_ns: int = field(default_factory=time.monotonic_ns)
    candidate_cutover: CandidateVersion | None = None
    inherited: Version | None = None


@dataclass(slots=True)
class RecordedEvent:
    event: Event
    version_id: str
    status: str
    action: Action | None = None

    def ack(self) -> EventAck:
        return {"event_id": self.event.event_id, "version_id": self.version_id,
                "status": self.action.status if self.action is not None else self.status,
                "permit": self.action.permit if self.action is not None else None}


class Directory:
    def __init__(self, model_revision: str, tokenizer_hash: str, layout_hash: str):
        self.runtime = (model_revision, tokenizer_hash, layout_hash)
        self.branches: dict[BranchKey, Branch] = {}
        self.versions: dict[str, Version] = {}
        self.actions: dict[str, Action] = {}
        self.rids: dict[str, Action] = {}
        self.events: dict[tuple[str, str], RecordedEvent] = {}
        self.fork_versions: dict[BranchKey, str] = {}
        self.discarded: list[tuple[Branch, CandidateVersion]] = []

    def branch(self, key: BranchKey) -> Branch:
        if key not in self.branches:
            raise unknown("unknown session/branch")
        return self.branches[key]

    def _new_version(self, messages: tuple[str, ...], surface: str | None, history: str,
                     parent: Version | None = None) -> Version:
        version = Version(uuid.uuid4().hex, messages, surface, history, parent=parent)
        self.versions[version.id] = version
        return version

    def salt(self, branch: Branch, version: Version, ids: tuple[int, ...]) -> str:
        if version.fork_prefix is not None:
            prefix = version.fork_prefix
            if not prefix or ids[:len(prefix)] != prefix:
                version.history = digest(branch.current.history, "fork-cold", version.id)
            version.fork_prefix = None
        return digest(*self.runtime, branch.run_id, branch.domain, version.history)

    def _discard(self, branch: Branch, candidate: CandidateVersion) -> None:
        if candidate.status != "canceled":
            candidate.status = "canceled"
            self.discarded.append((branch, candidate))
            action = self.actions.get(candidate.action_id)
            if action is not None and action.status not in TERMINAL:
                action.set_status("canceled", "candidate superseded")
        if branch.candidate is candidate:
            branch.candidate = None
        if branch.candidate_cutover is candidate:
            branch.candidate_cutover = None

    def _new_branch(self, event: Event, key: BranchKey) -> Branch:
        if event.expected_version is not None or event.seq != 1 or event.kind != EventKind.APPEND:
            raise unknown("branch must begin with append(seq=1, expected_version=null)")
        domain = identifier(event.payload.get("isolation_domain", event.session_id), "isolation_domain")
        version = self._new_version(self._messages(event), self._surface(event), "initial")
        branch = Branch(event.run_id, key, domain, version)
        self.branches[key] = branch
        return branch

    @staticmethod
    def _messages(event: Event) -> tuple[str, ...]:
        return _message_ids(event.payload.get("message_ids"))

    @staticmethod
    def _surface(event: Event) -> str | None:
        surface = event.payload.get("surface_hash")
        return None if surface is None else identifier(surface, "surface_hash")

    def accept(self, event: Event) -> RecordedEvent:
        event_key = (event.run_id, event.event_id)
        recorded = self.events.get(event_key)
        if recorded is not None:
            if recorded.event.identity != event.identity:
                raise conflict("event_id reused with different content")
            return recorded
        key = (event.session_id, event.branch_id)
        branch = self.branches.get(key)
        action = None
        status = "ok"
        if branch is None:
            branch = self._new_branch(event, key)
        else:
            if event.run_id != branch.run_id:
                raise unknown("unknown branch in run")
            if event.seq != branch.seq + 1 or event.expected_version != branch.current.id:
                raise conflict("branch sequence or committed version changed")
            status, action = self._apply(branch, event)
        branch.seq = event.seq
        branch.last_activity_ns = time.monotonic_ns()
        recorded = RecordedEvent(event, branch.current.id, status, action)
        self.events[event_key] = recorded
        return recorded

    def _apply(self, branch: Branch, event: Event) -> tuple[str, Action | None]:
        kind, payload = event.kind, event.payload
        if branch.lifecycle == "expired" and kind not in {EventKind.CANCEL, EventKind.END, EventKind.FINISH, EventKind.COMMIT}:
            raise conflict("branch already ended")
        if kind in {EventKind.APPEND, EventKind.REPLACE, EventKind.REMOVE}:
            self._change_view(branch, event)
        elif kind == EventKind.FORK:
            self._fork(branch, payload)
        elif kind == EventKind.READY:
            return "ready", self._ready(branch, payload)
        elif kind == EventKind.SUSPEND:
            if branch.lifecycle == "expired":
                raise conflict("branch expired")
            branch.lifecycle = "suspended"
        elif kind == EventKind.RESUME:
            if branch.lifecycle == "expired":
                raise conflict("branch expired")
            branch.lifecycle = "active"
        elif kind in {EventKind.END, EventKind.CANCEL}:
            action_id = payload.get("action_id") if kind == EventKind.CANCEL else None
            if action_id is not None:
                action = self._pick_action(branch, action_id)
                if action.status not in TERMINAL:
                    action.set_status("canceled", "canceled by harness")
                if action.status == "canceled" and branch.candidate is not None and branch.candidate.action_id == action.id:
                    self._discard(branch, branch.candidate)
                return action.status, None
            branch.lifecycle = "expired"
            if branch.candidate is not None:
                self._discard(branch, branch.candidate)
            if branch.candidate_cutover is not None:
                self._discard(branch, branch.candidate_cutover)
            for action_id in branch.action_ids:
                action = self.actions[action_id]
                if action.status not in TERMINAL:
                    action.set_status("canceled", "branch ended")
            return "completed" if kind == EventKind.END else "canceled", None
        elif kind in {EventKind.START, EventKind.FINISH, EventKind.COMMIT, EventKind.JOIN}:
            return self._settle(branch, kind, payload), None
        return "ok", None

    def _change_view(self, branch: Branch, event: Event) -> None:
        old = branch.current
        new_messages, surface = self._messages(event), self._surface(event)
        domain = event.payload.get("isolation_domain", branch.domain)
        if domain != branch.domain:
            raise conflict("isolation domain cannot change")
        if old.message_ids == new_messages and old.surface_hash == surface:
            if event.kind == EventKind.APPEND:
                return
        append = event.kind == EventKind.APPEND
        if append and new_messages[:len(old.message_ids)] != old.message_ids:
            # A child's first native prompt may differ from the inherited log.
            # In that case only a cold namespace is legal.
            if branch.seq != 0 or branch.inherited is None:
                raise conflict("append changed a frozen message identity")
            append = False
        candidate = branch.candidate
        if not append and candidate is not None:
            expected = candidate.expected_target
            approved = event.payload.get("candidate_action_id") == candidate.action_id
            if (approved and candidate.status == "ready" and
                    (expected is None or new_messages[:len(expected)] == expected) and
                    old.message_ids[:len(candidate.boundary)] == candidate.boundary):
                branch.candidate_cutover = candidate
                branch.candidate = None
            else:
                self._discard(branch, candidate)
        if not append and branch.candidate_cutover is not None and branch.candidate_cutover is not candidate:
            self._discard(branch, branch.candidate_cutover)
        history = old.history if append else digest(old.history, event.event_id, event.kind)
        branch.current = self._new_version(new_messages, surface, history, old)
        if append:
            branch.current.fork_prefix = old.fork_prefix

    def _fork(self, branch: Branch, payload: dict[str, Any]) -> None:
        parent = identifier(payload.get("parent_branch_id"), "parent_branch_id")
        parent_session = identifier(payload.get("parent_session_id", branch.key[0]), "parent_session_id")
        source = self.branch((parent_session, parent))
        if source is not branch or payload.get("isolation_domain", branch.domain) != branch.domain:
            raise conflict("fork must be declared by its parent in the same isolation domain")
        base = source.last_ended_turn
        if base is None or base.id != payload.get("parent_version"):
            raise conflict("fork requires the parent's last ended turn version")
        child_session = identifier(payload.get("child_session_id", branch.key[0]), "child_session_id")
        child_id = identifier(payload.get("child_branch_id"), "child_branch_id")
        child_key = (child_session, child_id)
        if child_key in self.branches:
            raise conflict("child branch already exists")
        version = self._new_version(base.message_ids, base.surface_hash, base.history, base)
        checkpoint = base
        while checkpoint.tokens is None and checkpoint.parent is not None and checkpoint.history == checkpoint.parent.history:
            checkpoint = checkpoint.parent
        version.fork_prefix = checkpoint.tokens or ()
        self.branches[child_key] = Branch(branch.run_id, child_key, branch.domain, version, inherited=checkpoint)
        self.fork_versions[child_key] = version.id

    def _ready(self, branch: Branch, payload: dict[str, Any]) -> Action:
        if branch.lifecycle != "active":
            raise conflict("branch is not active")
        action_id = identifier(payload.get("action_id"), "action_id")
        rid = identifier(payload.get("rid"), "rid")
        turn = identifier(payload.get("turn_id"), "turn_id")
        call = identifier(payload.get("call_id"), "call_id")
        kind, ordinal = payload.get("action_kind"), payload.get("ordinal")
        if kind not in ("model", "tool", "child", "summary", "prepare", "batch"):
            raise invalid("unknown action_kind")
        if type(ordinal) is not int or not 0 <= ordinal < 2**53:
            raise invalid("invalid action ordinal")
        if action_id in self.actions or rid in self.rids:
            raise conflict("action_id or rid already used")
        dependencies = payload.get("dependencies", [])
        if not isinstance(dependencies, list):
            raise invalid("dependencies must be an array")
        dependencies = tuple(identifier(item, "dependency") for item in dependencies)
        for dep in dependencies:
            if dep not in self.actions or self.actions[dep].run_id != branch.run_id:
                raise unknown("unknown action dependency")
        if "wait_type" in payload:
            identifier(payload["wait_type"], "wait_type")
        if "oracle_wait_ms" in payload:
            oracle_wait = payload["oracle_wait_ms"]
            if type(oracle_wait) not in (int, float) or not 0 <= oracle_wait <= 2000:
                raise invalid("oracle_wait_ms must be in [0, 2000]")
        if kind == "prepare":
            if branch.candidate is not None or branch.candidate_cutover is not None:
                raise conflict("branch already has a candidate")
            boundary = _message_ids(payload.get("boundary_message_ids"))
            target_raw = payload.get("target_message_ids")
            target = None if target_raw is None else _message_ids(target_raw)
            if not boundary or branch.current.message_ids[:len(boundary)] != boundary:
                raise conflict("candidate did not freeze a committed message boundary")
            if not isinstance(payload.get("chat"), dict):
                raise invalid("prepare requires exact native chat payload")
            prepared = self._new_version(target or (), None,
                                         digest(branch.current.history, "candidate", action_id), branch.current)
            branch.candidate = CandidateVersion(action_id, branch.current, boundary, target, prepared)
        now = time.monotonic_ns()
        action = Action(action_id, rid, branch.run_id, branch.key, branch.current.id, kind, turn,
                        call, ordinal, now, now + DEADLINE_NS, dependencies, payload)
        self.actions[action_id] = self.rids[rid] = action
        branch.action_ids.add(action_id)
        return action

    def _pick_action(self, branch: Branch, raw: Any) -> Action:
        aid = identifier(raw, "action_id")
        if aid not in branch.action_ids:
            raise unknown("unknown branch action")
        return self.actions[aid]

    def _settle(self, branch: Branch, kind: EventKind, payload: dict[str, Any]) -> str:
        raw = payload.get("action_id")
        if payload.get("turn_end") is True:
            identifier(payload.get("turn_id"), "turn_id")
        if raw is None:
            if kind == EventKind.COMMIT and payload.get("turn_end") is True:
                branch.last_ended_turn = branch.current
                return "committed"
            if kind == EventKind.JOIN:
                children = payload.get("children")
                if not isinstance(children, list):
                    raise invalid("join requires declared child actions")
                for aid in children:
                    child = self._pick_action(branch, aid)
                    if child.status not in TERMINAL or child.draining:
                        raise conflict("join before a declared child reached terminal state")
                return "committed"
            raise invalid("missing action_id")
        action = self._pick_action(branch, raw)
        if kind == EventKind.START:
            self.start(action, branch, consume=False)
        elif kind == EventKind.FINISH:
            status = payload.get("status", "completed")
            if status not in ("ok", "completed", "canceled", "error", "failed", "success"):
                raise invalid("invalid action finish status")
            if action.status not in ("running", "completed", "canceled", "error", "committed"):
                raise conflict("finish before dispatch")
            if (action.kind in ("model", "summary", "batch", "prepare") and action.status == "running"
                    and status in ("ok", "completed", "success")):
                raise conflict("native terminal acknowledgement has not arrived")
            action.draining = False
            if action.status == "running" or (action.status == "completed" and status in ("failed", "error", "canceled")):
                action.set_status("error" if status in ("failed", "error") else
                                  "canceled" if status == "canceled" else "completed")
        elif kind == EventKind.COMMIT:
            if action.status not in TERMINAL or action.draining:
                raise conflict("commit before action terminal state")
            action.committed = True
            if action.status == "completed":
                action.set_status("committed")
            if payload.get("turn_end") is True:
                branch.last_ended_turn = branch.current
        return action.status

    def grant(self, action: Action, expires: int) -> None:
        action.permit = {"action_id": action.id, "version_id": action.version_id, "expires_mono_ns": expires}
        action.set_status("granted")

    def dispatchable(self, action: Action, branch: Branch) -> bool:
        if branch.lifecycle != "active":
            return False
        if branch.current.id == action.version_id:
            return True
        candidate = branch.candidate
        return (action.kind == "prepare" and candidate is not None and candidate.action_id == action.id
                and branch.current.history == candidate.base.history
                and branch.current.message_ids[:len(candidate.boundary)] == candidate.boundary)

    def start(self, action: Action, branch: Branch, *, consume: bool = True) -> None:
        if action.status != "granted" or action.permit is None or not (
            self.dispatchable(action, branch) and
            action.permit["expires_mono_ns"] > time.monotonic_ns()
        ):
            raise conflict("action permit expired, canceled, or does not match committed version")
        if action.kind in {"model", "batch", "summary"}:
            if branch.owner not in (None, action.id):
                raise conflict("branch already has foreground generation")
            branch.owner = action.id
        if consume or action.kind not in {"model", "batch", "summary"}:
            if action.kind in ("tool", "child"):
                action.draining = True
            action.set_status("running")

    def settle_engine(self, action: Action, status: str, error: str | None = None) -> None:
        branch = self.branches[action.key]
        if branch.owner == action.id:
            branch.owner = None
        if action.status not in TERMINAL:
            action.set_status(status, error)

    def lose_epoch(self) -> None:
        for version in self.versions.values():
            version.handle = None
            version.residency = "absent"
            version.prepare_lease = False
        for branch in self.branches.values():
            if branch.candidate is not None:
                self._discard(branch, branch.candidate)
            if branch.candidate_cutover is not None:
                self._discard(branch, branch.candidate_cutover)
            branch.owner = None
        for action in self.actions.values():
            if action.status not in TERMINAL:
                action.set_status("error", "scheduler epoch lost; action was not retried")
