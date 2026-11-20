"""Scheduler-thread ownership of AgenticLayer bindings and native radix leases.

All methods of EngineCache run on the SGLang scheduler thread. Socket workers
only exchange immutable messages with that thread; they never access CUDA or
SGLang's allocator and tree structures.
"""
from __future__ import annotations

from array import array
from dataclasses import dataclass, field
import time
import uuid
from typing import Any

from .protocol import MAX_CONTEXT, MAX_GENERATE, canonical_json, identifier, token_ids


@dataclass
class Binding:
    command: dict[str, Any]
    ids: tuple[int, ...]
    salt: str
    kind: str
    hold: bool
    session: str
    branch: str
    max_new_tokens: int
    reserved_prefix: int = 0
    bind_lock: Any = None
    req: Any = None
    accepted: bool = False
    canceled: bool = False
    input_ids: list[int] = field(default_factory=list)
    matched: int = 0
    full_hit: int = 0
    match_observed: bool = False


@dataclass
class Handle:
    version: str
    ids: tuple[int, ...]
    salt: str
    node: Any
    device_lock: Any = None
    host_receipt: Any = None
    host_node: Any = None
    host_leases: int = 0
    leases: int = 0
    inflight: int = 0


@dataclass
class Transfer:
    command: dict[str, Any]
    handle: str
    nodes: set[Any]
    started_ns: int
    kind: str
    canceled: bool = False
    transfer_bytes: int = 0
    transfer_us: int = 0
    metrics_complete: bool = True
    error: str | None = None
    expected_nodes: tuple[Any, ...] = ()
    lock: Any = None
    acked_nodes: set[Any] = field(default_factory=set)


class EngineCache:
    def __init__(self, scheduler: Any):
        self.scheduler = scheduler
        self.cache = scheduler.tree_cache
        self.epoch = uuid.uuid4().hex
        self.seq = 0
        self.bindings: dict[str, Binding] = {}
        self.used_rids: set[str] = set()
        self.handles: dict[str, Handle] = {}
        self.transfers: dict[str, Transfer] = {}
        self.actions: dict[str, tuple[bytes, dict[str, Any]]] = {}
        self.completions: list[dict[str, Any]] = []
        self.cache._agentic_engine = self

    def _match(self, ids: tuple[int, ...], salt: str):
        if any(t.kind == 'restore' and self.handles[t.handle].salt == salt
               for t in self.transfers.values()):
            raise ValueError('namespace has an unacknowledged host restore')
        from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        return self.cache.match_prefix(MatchPrefixParams(
            key=RadixKey(array('q', ids), cache_salt=salt), cow_mamba=False))

    @staticmethod
    def _pool(allocator: Any) -> dict[str, int]:
        if allocator is None:
            return dict(capacity=0, free=0, owned=0, reserved=0)
        capacity = int(getattr(allocator, 'logical_size', allocator.size))
        free = int(allocator.available_size())
        return dict(capacity=capacity, free=free, owned=capacity - free, reserved=0)

    def pools(self) -> dict[str, dict[str, int]]:
        c = self.cache
        full = self._pool(c.token_to_kv_pool_allocator)
        recurrent = self._pool(c.req_to_token_pool.mamba_allocator)
        group = c.host_pool_group
        host = self._pool(group.anchor_entry.host_pool if group else None)
        for b in self.bindings.values():
            if b.canceled and not b.accepted:
                continue
            kv = b.req.kv if b.req is not None else None
            materialized = max(0, kv.kv_allocated_len - b.reserved_prefix) if kv is not None else 0
            full['reserved'] += max(0, len(b.ids) - b.reserved_prefix + b.max_new_tokens - materialized)
            recurrent['reserved'] += int(kv is None or kv.mamba_pool_idx is None)
        return dict(full=full, recurrent=recurrent, host=host)

    def _slot_bytes(self) -> tuple[int, int]:
        c = self.cache
        full_pool = c.token_to_kv_pool_allocator.get_kvcache().full_kv_pool
        full_bytes = sum(t.numel() * t.element_size() // t.shape[0]
                         for t in (*full_pool.k_buffer, *full_pool.v_buffer))
        state = c.req_to_token_pool.mamba_pool.mamba_cache
        recurrent_bytes = sum(t.numel() * t.element_size() // t.shape[1]
                              for t in (*state.conv, state.temporal))
        return full_bytes, recurrent_bytes

    def _bytes(self, pools: dict[str, dict[str, int]]) -> tuple[int, int]:
        c = self.cache
        full_bytes, recurrent_bytes = self._slot_bytes()
        host_bytes = 0
        if c.host_pool_group:
            for entry in c.host_pool_group.entries:
                pool = entry.host_pool
                host_bytes += (getattr(pool, 'logical_size', pool.size) - pool.available_size()) * pool.size_per_token
        return (pools['full']['owned'] * full_bytes +
                pools['recurrent']['owned'] * recurrent_bytes, host_bytes)

    def _reply(self, command: dict[str, Any], status: str, *, error: str | None = None,
               handle: str | None = None, matched: int = 0, details: dict | None = None,
               transfer_us: int = 0, transfer_bytes: int = 0) -> dict[str, Any]:
        self.seq += 1
        pools = self.pools()
        hbm_bytes, host_bytes = self._bytes(pools)
        out = dict(epoch=self.epoch, seq=self.seq, action_id=command.get('action_id', 'snapshot'),
                   status=status, version_id=command.get('version_id', 'snapshot'),
                   matched_tokens=matched, full_kv_slots=pools['full']['owned'],
                   recurrent_slots=pools['recurrent']['owned'], unique_hbm_bytes=hbm_bytes,
                   unique_host_bytes=host_bytes, reserved_bytes=(
                       pools['full']['reserved'] * self._slot_bytes()[0] +
                       pools['recurrent']['reserved'] * self._slot_bytes()[1]),
                   fg_queue=len(self.scheduler.waiting_queue), bg_queue=len(self.transfers),
                   transfer_bytes=transfer_bytes, transfer_us=transfer_us, error=error,
                   run_id=command.get('run_id', 'snapshot'), rid=command.get('rid', 'snapshot'),
                   handle=handle)
        if details is not None:
            out['details'] = details
        return out

    def snapshot(self, command: dict[str, Any], after: int | None = None) -> dict[str, Any]:
        # A frame can include a full 32768-token input. Limit by encoded size,
        # not by count, so the next receipt is always available on another page.
        completions: list[dict[str, Any]] = []
        if after is not None:
            if isinstance(after, bool) or not isinstance(after, int) or after < 0:
                raise ValueError('after_seq must be a nonnegative integer')
            for receipt in self.completions:
                if receipt['seq'] > after:
                    if completions and len(canonical_json(completions)) + len(canonical_json(receipt)) > 700_000:
                        break
                    completions.append(receipt)
        from .kernels import transfer_counters
        pools = self.pools()
        full_bytes, recurrent_bytes = self._slot_bytes()
        host_capacity = 0
        if self.cache.host_pool_group:
            host_capacity = sum(getattr(e.host_pool, 'logical_size', e.host_pool.size) *
                                e.host_pool.size_per_token for e in self.cache.host_pool_group.entries)
        details = dict(protocol=1, capabilities=dict(bind=True, lease=True, release=True, repos=True,
                       prepare=True, offload=self.cache.cache_controller is not None,
                       restore=self.cache.cache_controller is not None, reclaim=True,
                       completion_paging=True), pools=pools, completions=completions,
                       full_slot_bytes=full_bytes, recurrent_slot_bytes=recurrent_bytes,
                       hbm_capacity_bytes=pools['full']['capacity'] * full_bytes +
                           pools['recurrent']['capacity'] * recurrent_bytes,
                       host_capacity_bytes=host_capacity,
                       hbm_reserved_bytes=pools['full']['reserved'] * full_bytes +
                           pools['recurrent']['reserved'] * recurrent_bytes,
                       host_reserved_bytes=0, transfer_counters=transfer_counters())
        return self._reply(command, 'ok', details=details)

    def execute(self, command: dict[str, Any]) -> dict[str, Any]:
        op = command['op']
        if op == 'snapshot':
            return self.snapshot(command, command.get('payload', {}).get('after_seq'))
        fingerprint = canonical_json(command)
        old = self.actions.get(command['action_id'])
        if old:
            if old[0] != fingerprint:
                return self._reply(command, 'error', error='action_id reused with different content')
            return old[1]
        try:
            method = getattr(self, '_op_' + op)
            result = method(command)
        except (ValueError, KeyError, TypeError) as exc:
            result = self._reply(command, 'error', error=str(exc))
        self.actions[command['action_id']] = (fingerprint, result)
        return result

    def _op_bind(self, cmd: dict[str, Any]) -> dict[str, Any]:
        p = cmd.get('payload', {})
        ids = token_ids(p.get('expected_ids'))
        salt = p.get('cache_salt')
        if not isinstance(salt, str) or len(salt) != 64 or any(c not in '0123456789abcdef' for c in salt):
            raise ValueError('cache_salt must be a lowercase SHA-256 digest')
        kind = p.get('kind')
        if kind not in ('chat', 'batch', 'prepare'):
            raise ValueError('invalid bind kind')
        n = p.get('max_new_tokens')
        if isinstance(n, bool) or not isinstance(n, int) or not 0 <= n <= MAX_GENERATE or len(ids) + n > MAX_CONTEXT:
            raise ValueError('invalid context or generation length')
        hold = p.get('hold', False)
        if not isinstance(hold, bool) or (hold and kind != 'prepare'):
            raise ValueError('hold is only valid for prepare')
        session = identifier(p.get('session_id'), 'session_id')
        branch = identifier(p.get('branch_id'), 'branch_id')
        if cmd['rid'] in self.used_rids:
            raise ValueError('rid already bound')
        if any((not b.canceled or b.accepted) and b.session == session and b.branch == branch for b in self.bindings.values()):
            raise ValueError('branch already has an active request')
        if any(t.kind == 'restore' and self.handles[t.handle].salt == salt
               for t in self.transfers.values()):
            return self._reply(cmd, 'unavailable', error='namespace has an unacknowledged host restore')
        match = self._match(ids[:-1], salt)
        prefix = match.device_prefix_len
        bind_lock = self.cache.lock(match.last_device_node) if prefix else None
        try:
            pools = self.pools()
            required_full = pools['full']['reserved'] + len(ids) - prefix + n
            required_recurrent = pools['recurrent']['reserved'] + 1
            if required_full > pools['full']['capacity'] or required_recurrent > pools['recurrent']['capacity']:
                return self._reply(cmd, 'unavailable', error='insufficient arena capacity')
            full_deficit = max(0, required_full - pools['full']['free'])
            recurrent_deficit = max(0, required_recurrent - pools['recurrent']['free'])
            if full_deficit or recurrent_deficit:
                from sglang.srt.mem_cache.base_prefix_cache import EvictParams
                self.cache.evict(EvictParams(num_tokens=full_deficit, mamba_num=recurrent_deficit))
                pools = self.pools()
                if required_full > pools['full']['free'] or required_recurrent > pools['recurrent']['free']:
                    return self._reply(cmd, 'unavailable', error='insufficient reclaimable capacity')
            self.bindings[cmd['rid']] = Binding(cmd, ids, salt, kind, hold, session, branch, n,
                                                reserved_prefix=prefix, bind_lock=bind_lock)
            bind_lock = None
            self.used_rids.add(cmd['rid'])
            return self._reply(cmd, 'ok')
        finally:
            if bind_lock is not None:
                self.cache.unlock(bind_lock)

    def _handle(self, cmd: dict[str, Any]) -> Handle:
        h = self.handles.get(cmd.get('handle'))
        if h is None:
            raise ValueError('unknown handle')
        if h.version != cmd['version_id']:
            raise ValueError('handle version mismatch')
        return h

    def _op_lease(self, cmd: dict[str, Any]) -> dict[str, Any]:
        h = self._handle(cmd)
        match = self._match(h.ids, h.salt)
        residency = cmd.get('payload', {}).get('residency', 'hbm')
        if residency == 'host':
            from sglang.srt.mem_cache.unified_cache.components import ComponentType
            if self.cache.cache_controller is None or match.full_kv_hit_length < len(h.ids):
                return self._reply(cmd, 'unavailable', error='host checkpoint is unavailable')
            node_id = match.best_match_node
            node = self.cache.tree_core.node_by_id(node_id)
            state = node.component_data[ComponentType.MAMBA]
            if not node.backuped or state.host_value is None:
                return self._reply(cmd, 'unavailable', error='complete host hybrid checkpoint is unavailable')
            if h.host_leases == 0:
                h.host_node = node_id
                h.host_receipt = self.cache.inc_host_lock_ref(node_id).to_dec_params()
            h.host_leases += 1
            return self._reply(cmd, 'ok', handle=cmd['handle'],
                               matched=len(h.ids), details={'residency': 'host'})
        if residency != 'hbm':
            raise ValueError('invalid lease residency')
        if match.device_prefix_len != len(h.ids):
            return self._reply(cmd, 'unavailable', error='full hybrid checkpoint is not device resident')
        if h.leases == 0:
            h.node = match.last_device_node
            h.device_lock = self.cache.lock(h.node)
        h.leases += 1
        return self._reply(cmd, 'ok', handle=cmd['handle'], matched=match.device_prefix_len,
                           details={'residency': 'hbm'})

    def _op_release(self, cmd: dict[str, Any]) -> dict[str, Any]:
        h = self._handle(cmd)
        residency = cmd.get('payload', {}).get('residency', 'hbm')
        if residency == 'host':
            if not h.host_leases:
                raise ValueError('handle has no host lease to release')
            h.host_leases -= 1
            if h.host_leases == 0:
                self.cache.dec_host_lock_ref(h.host_node, h.host_receipt)
                h.host_receipt = h.host_node = None
        elif residency == 'hbm':
            if not h.leases:
                raise ValueError('handle has no device lease to release')
            h.leases -= 1
            if h.leases == 0:
                self.cache.unlock(h.device_lock)
                h.device_lock = None
        else:
            raise ValueError('invalid lease residency')
        return self._reply(cmd, 'ok', handle=cmd['handle'], details={'residency': residency})

    def _op_cancel(self, cmd: dict[str, Any]) -> dict[str, Any]:
        binding = self.bindings.get(cmd['rid'])
        if binding is not None:
            if binding.canceled:
                return self._reply(cmd, 'canceled')
            binding.canceled = True
            if binding.accepted:
                from sglang.srt.managers.io_struct import AbortReq
                self.scheduler.abort_request(AbortReq(rid=cmd['rid']))
            else:
                self._release_bind_lock(binding)
                self.completions.append(self._reply(binding.command, 'canceled',
                    details=dict(token_ids=[], full_kv_hit_length=0, residency='absent')))
            return self._reply(cmd, 'accepted' if binding.accepted else 'canceled')
        target = cmd.get('payload', {}).get('action_id')
        transfer = self.transfers.get(target) if target else next(
            (t for t in self.transfers.values() if t.command['rid'] == cmd['rid']), None)
        if transfer:
            transfer.canceled = True
            return self._reply(cmd, 'accepted', handle=cmd['handle'])
        for receipt in reversed(self.completions):
            if receipt.get('rid') == cmd['rid']:
                return self._reply(cmd, receipt['status'], handle=receipt.get('handle'),
                                   details=receipt.get('details'))
        return self._reply(cmd, 'not_found', error='no active request or transfer')

    def _op_offload(self, cmd: dict[str, Any]) -> dict[str, Any]:
        h = self._handle(cmd)
        if self.cache.cache_controller is None:
            return self._reply(cmd, 'unavailable', error='HiCache is disabled')
        if h.inflight:
            return self._reply(cmd, 'unavailable', error='transfer already in flight')
        match = self._match(h.ids, h.salt)
        if match.device_prefix_len != len(h.ids):
            return self._reply(cmd, 'unavailable', error='checkpoint is not fully resident')
        # Pin the source until the native D2H completion has been consumed.
        lock = self.cache.lock(match.last_device_node)
        action = self.cache.tree_core.build_backup_kv_action(match.last_device_node)
        before = set(self.cache.ongoing_write_through)
        failure = None
        try:
            self.cache._execute_and_commit_kv_backup(action, write_back=True)
        except Exception as exc:
            failure = str(exc)
        nodes = set(self.cache.ongoing_write_through) - before
        if not nodes:
            self.cache.unlock(lock)
            return self._reply(cmd, 'error' if failure else 'unavailable',
                               error=failure or 'no new backup was submitted')
        h.inflight += 1
        transfer = Transfer(cmd, cmd['handle'], nodes, time.monotonic_ns(), 'offload',
                            error=failure, expected_nodes=tuple(action.node_ids), lock=lock)
        self.transfers[cmd['action_id']] = transfer
        try:
            self.cache.flush_pending_backups()
        except Exception as exc:
            # A partially submitted copy may still read the source. Keep its
            # lock and in-flight record, and never automatically retry it.
            transfer.error = str(exc)
            return self._reply(cmd, 'error', error=str(exc), handle=cmd['handle'])
        return self._reply(cmd, 'accepted', handle=cmd['handle'])

    def _op_restore(self, cmd: dict[str, Any]) -> dict[str, Any]:
        h = self._handle(cmd)
        if self.cache.cache_controller is None or h.inflight:
            return self._reply(cmd, 'unavailable', error='HiCache unavailable or transfer in flight')
        if any(b.salt == h.salt and (not b.canceled or b.accepted) for b in self.bindings.values()):
            return self._reply(cmd, 'unavailable', error='same namespace has a queued or active forward')
        m = self._match(h.ids, h.salt)
        if m.device_prefix_len == len(h.ids):
            return self._reply(cmd, 'ready', handle=cmd['handle'], matched=m.device_prefix_len)
        if not (m.host_hit_length or m.mamba_host_hit_length) or not self.cache.has_free_load_back_slot():
            return self._reply(cmd, 'unavailable', error='no valid host checkpoint or load slot')
        pools = self.pools()
        if (pools['full']['free'] - pools['full']['reserved'] < m.host_hit_length or
                pools['recurrent']['free'] - pools['recurrent']['reserved'] < m.mamba_host_hit_length):
            return self._reply(cmd, 'unavailable', error='restore would consume foreground reservations')
        if not self.cache.load_back(m.best_match_node, mem_quota=None, req=None):
            return self._reply(cmd, 'unavailable', error='native load-back declined')
        self.cache.ready_to_load_host_cache()
        h.inflight += 1
        self.transfers[cmd['action_id']] = Transfer(cmd, cmd['handle'], {m.best_match_node}, time.monotonic_ns(), 'restore')
        return self._reply(cmd, 'accepted', handle=cmd['handle'])

    def _op_reclaim(self, cmd: dict[str, Any]) -> dict[str, Any]:
        h = self._handle(cmd)
        if h.inflight:
            return self._reply(cmd, 'unavailable', error='handle has an in-flight transfer')
        m = self._match(h.ids, h.salt)
        core = self.cache.tree_core
        payload = cmd.get('payload', {})
        residency = payload.get('residency', 'hbm')
        before = self.pools()
        if residency == 'host':
            if h.leases or h.host_leases or m.full_kv_hit_length != len(h.ids):
                return self._reply(cmd, 'unavailable', error='host checkpoint is leased or incomplete')
            node = core.node_by_id(m.best_match_node)
            if not core._is_host_leaf(node):
                return self._reply(cmd, 'unavailable', error='checkpoint is not an idle host leaf')
            from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import DriveHostEvictionResult
            result = DriveHostEvictionResult()
            try:
                core._evict_host_leaf(node, result.tracker, result.device_frees,
                                      result.host_frees, result.cache_actions)
            finally:
                self.cache._apply_eviction_result(result)
            after = self.pools()
            return self._reply(cmd, 'completed', handle=cmd['handle'], details={
                'freed_host_slots': before['host']['owned'] - after['host']['owned'],
                'residency': 'absent'})
        if residency != 'hbm':
            raise ValueError('invalid reclaim residency')
        if h.leases or m.device_prefix_len != len(h.ids):
            return self._reply(cmd, 'unavailable', error='device checkpoint is leased or incomplete')
        node = core.node_by_id(m.last_device_node)
        if (not core._is_device_leaf(node) or node.write_through_pending_id is not None
                or node.load_back_pending_id is not None):
            return self._reply(cmd, 'unavailable', error='checkpoint is not an idle device leaf')
        backed = node.backuped
        from collections import defaultdict
        tracker = defaultdict(int)
        if self.cache.cache_controller is not None and not backed:
            # Explicit expiry may discard an isolated, unbacked leaf. Do not
            # convert reclaim into D2H or discard any descendant or host data.
            if (payload.get('discard_unbacked') is not True or node.children
                    or h.host_leases or any(cd.host_value is not None or cd.host_lock_ref
                                           for cd in node.component_data)):
                return self._reply(cmd, 'unavailable', error='checkpoint has unbacked component state')
            result = core.drop_subtree_no_host(node.id)
            self.cache._apply_eviction_result(result)
            if not result.is_dropped:
                return self._reply(cmd, 'unavailable', error='unbacked checkpoint is not safely reclaimable')
        elif (self.cache.cache_controller is not None
                and any(cd.value is not None and cd.host_value is None for cd in node.component_data)):
            return self._reply(cmd, 'unavailable', error='partially backed checkpoint cannot be demoted')
        elif self.cache._evict_device_leaf(node.id, tracker) is not None:
            raise RuntimeError('idle device leaf unexpectedly requested backup')
        after = self.pools()
        return self._reply(cmd, 'completed', handle=cmd['handle'], details={
            'freed_full_slots': before['full']['owned'] - after['full']['owned'],
            'freed_recurrent_slots': before['recurrent']['owned'] - after['recurrent']['owned'],
            'residency': 'host' if backed else 'absent'})

    def _rope(self) -> tuple[int, float]:
        text = self.scheduler.model_config.hf_text_config
        head_dim = int(getattr(text, 'head_dim', 0) or 0)
        params = getattr(text, 'rope_parameters', None) or {}
        theta = float(params.get('rope_theta', getattr(text, 'rope_theta', 1_000_000.0)))
        partial = float(params.get('partial_rotary_factor', getattr(text, 'partial_rotary_factor', 1.0)))
        rotary_dim = int(head_dim * partial)
        if rotary_dim <= 0 or rotary_dim % 2:
            raise RuntimeError('model has no even partial rotary dimension')
        return rotary_dim, theta

    def _chain_slots(self, handle: Handle, component_type: Any):
        import torch
        node = self.cache.tree_core.node_by_id(handle.node)
        chunks = []
        while node is not None and node.parent is not None:
            value = node.component_data[component_type].value
            if value is None:
                return None
            chunks.append(value)
            node = node.parent
        if not chunks:
            return None
        return torch.cat(list(reversed(chunks)))

    def _op_repos(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Materialise the surviving KV of ``handle`` at compacted positions.

        Drops token spans from the visible sequence while every retained key
        keeps its computed value: only RoPE is re-applied, so no re-prefill is
        paid for the suffix. The result is inserted under a fresh cache salt,
        and the next bind for the compacted sequence matches it natively.
        """
        h = self.handles.get(cmd.get('handle') or '')
        if h is None:
            raise ValueError('unknown handle')
        payload = cmd.get('payload') or {}
        s = payload.get('cache_salt')
        if not isinstance(s, str) or len(s) != 64 or any(c not in '0123456789abcdef' for c in s):
            raise ValueError('cache_salt must be a lowercase SHA-256 digest')
        if s == h.salt:
            raise ValueError('repos requires a fresh cache salt')
        ids = token_ids(payload.get('expected_ids'))
        spans = payload.get('dropped')
        if not isinstance(spans, list) or not spans:
            raise ValueError('repos requires dropped spans')
        spans = sorted((int(a), int(b)) for a, b in spans)
        cursor = 0
        for start, end in spans:
            if start < cursor or end <= start or end > len(h.ids):
                raise ValueError('dropped spans must be sorted, disjoint and in range')
            cursor = end
        keep = [i for i in range(len(h.ids)) if not any(a <= i < b for a, b in spans)]
        if len(keep) != len(ids) or tuple(h.ids[i] for i in keep) != ids:
            raise ValueError('compacted ids do not match the source minus dropped spans')
        import torch
        import time as _time
        phases: dict[str, float] = {}
        t0 = _time.monotonic()

        def _tick(name: str) -> None:
            phases[name] = round((_time.monotonic() - t0) * 1000, 3)

        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_cache.components import ComponentType
        match = self._match(h.ids, h.salt)
        _tick('match_source')
        if match.device_prefix_len != len(h.ids):
            return self._reply(cmd, 'unavailable',
                               error=f'source chain was evicted ({match.device_prefix_len} of {len(h.ids)})')
        core = self.cache.tree_core
        source = core.node_by_id(h.node)
        src_slots = self._chain_slots(h, ComponentType.FULL)
        src_state = source.component_data[ComponentType.MAMBA].value
        _tick('chain_slots')
        if src_slots is None or len(src_slots) != len(h.ids):
            have = 'missing' if src_slots is None else str(len(src_slots))
            return self._reply(cmd, 'unavailable',
                               error=f'source KV is not fully device resident ({have} of {len(h.ids)})')
        if src_state is None:
            return self._reply(cmd, 'unavailable', error='source has no recurrent checkpoint')
        n = len(ids)
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams
        allocator = self.cache.token_to_kv_pool_allocator
        dst = allocator.alloc(n)
        if dst is None:
            # Reclaim stale namespaces like a bind does; the source chain is the
            # freshest insert, so LRU eviction cannot take it.
            self.cache.evict(EvictParams(num_tokens=n))
            dst = allocator.alloc(n)
            if dst is None:
                return self._reply(cmd, 'unavailable', error='insufficient arena capacity')
        mamba_allocator = self.cache.req_to_token_pool.mamba_allocator
        dst_state = mamba_allocator.alloc(1)
        if dst_state is None:
            self.cache.evict(EvictParams(mamba_num=1))
            dst_state = mamba_allocator.alloc(1)
            if dst_state is None:
                allocator.free(dst)
                return self._reply(cmd, 'unavailable', error='insufficient recurrent capacity')
        _tick('alloc')
        inserted = False
        try:
            full_pool = allocator.get_kvcache().full_kv_pool
            rotary_dim, theta = self._rope()
            from .kernels import copy_state_rows, rotate_kv
            old = torch.tensor(keep, dtype=torch.int64, device=dst.device)
            rotate_kv(src_slots[old], dst, full_pool.k_buffer, full_pool.v_buffer,
                      old, rotary_dim=rotary_dim, theta=theta)
            state = self.cache.req_to_token_pool.mamba_pool.mamba_cache
            copy_state_rows(int(src_state.reshape(-1)[0]), int(dst_state.reshape(-1)[0]),
                            state.conv, state.temporal)
            _tick('rotate')
            result = self.cache.insert(InsertParams(
                key=RadixKey(array('q', ids), cache_salt=s), value=dst,
                mamba_value=dst_state, inserted_len=0))
            _tick('insert')
            inserted = True
        except Exception:
            if inserted:
                raise
            allocator.free(dst)
            mamba_allocator.free(dst_state)
            raise
        # The post-insert match is required: its walk participates in the
        # native evictable-accounting ledgers the idle invariant checks.
        matched = self._match(ids, s)
        _tick('match_result')
        if matched.device_prefix_len != n or matched.last_device_node is None:
            allocator.free(dst)
            mamba_allocator.free(dst_state)
            return self._reply(cmd, 'unavailable', error='materialised chain is not device resident')
        node = matched.last_device_node
        handle = uuid.uuid4().hex
        # No lock: the materialised chain is a normal evictable tree node, so
        # the cache's own path-state cleanup stays authoritative.
        self.handles[handle] = Handle(version=cmd['version_id'], ids=ids, salt=s, node=node,
                                      device_lock=None, leases=1)
        slot_bytes, _ = self._slot_bytes()
        return self._reply(cmd, 'completed', handle=handle, matched=n, details={
            'source_tokens': len(h.ids), 'kept_tokens': n,
            'dropped_tokens': len(h.ids) - n, 'rotated_bytes': n * slot_bytes,
            'rotary_dim': rotary_dim, 'phases_ms': phases})

    def observe_acks(self, kind: str, acks: list[Any]) -> None:
        for ack in acks:
            nodes = set(ack.node_ids)
            owners = [t for t in self.transfers.values() if t.kind == kind and t.nodes & nodes]
            if len(owners) != 1 or not nodes.issubset(owners[0].nodes):
                for transfer in owners:
                    transfer.metrics_complete = False
                continue
            transfer = owners[0]
            transfer.acked_nodes.update(nodes)
            transfer.transfer_bytes += int(ack.num_bytes)
            if ack.timing_enabled:
                transfer.transfer_us += int(ack.start_event.elapsed_time(ack.finish_event) * 1000)
            else:
                transfer.metrics_complete = False

    def poll_transfers(self) -> None:
        for action, transfer in list(self.transfers.items()):
            pending = self.cache.ongoing_write_through if transfer.kind == 'offload' else self.cache.ongoing_load_back
            if any(node in pending for node in transfer.nodes):
                continue
            if not transfer.nodes.issubset(transfer.acked_nodes):
                transfer.metrics_complete = False
            h = self.handles[transfer.handle]
            if transfer.kind == 'offload':
                self.cache.unlock(transfer.lock)
            h.inflight -= 1
            del self.transfers[action]
            match = self._match(h.ids, h.salt)
            status = ('canceled' if transfer.canceled else
                      'error' if transfer.error else
                      'ready' if transfer.kind == 'restore' and match.device_prefix_len == len(h.ids) else
                      'partial' if transfer.kind == 'restore' else 'completed')
            if transfer.kind == 'offload' and status == 'completed':
                try:
                    for node in transfer.expected_nodes:
                        device_values, component_transfers = self.cache.tree_core.build_backup_spec(node)
                        if device_values.numel() or component_transfers:
                            status = 'partial'
                            break
                except (KeyError, ValueError, AssertionError):
                    # A split or eviction invalidated the old node identity.
                    status = 'partial'
            demoted = False
            if transfer.kind == 'offload' and status == 'completed':
                try:
                    outcome = self._op_reclaim(transfer.command)
                    demoted = outcome['status'] == 'completed'
                    if demoted:
                        match = self._match(h.ids, h.salt)
                except Exception as exc:
                    transfer.error = str(exc)
                    status = 'error'
            result = self._reply(transfer.command, status, handle=transfer.handle,
                                 error=transfer.error,
                                 matched=match.device_prefix_len,
                                 transfer_us=transfer.transfer_us,
                                 transfer_bytes=transfer.transfer_bytes,
                                 details=dict(token_ids=list(h.ids), full_kv_hit_length=match.full_kv_hit_length,
                                              transfer_metrics_complete=transfer.metrics_complete, demoted=demoted,
                                              residency='host' if demoted else ('hbm' if match.device_prefix_len == len(h.ids) else 'host')))
            self.completions.append(result)

    def accept(self, recv_req: Any) -> str | None:
        b = self.bindings.get(recv_req.rid)
        if b is None:
            # Probes are unbound, but a consumed control rid can never become
            # an uncontrolled request after cancellation or completion.
            return 'request binding was already consumed' if recv_req.rid in self.used_rids else None
        if b.accepted or b.canceled:
            return 'request binding is already consumed or canceled'
        actual = list(recv_req.input_ids or [])
        if tuple(actual) != b.ids or recv_req.cache_salt != b.salt or recv_req.input_embeds is not None:
            self.fail_binding(b, actual, 'authoritative tokens or namespace differ from binding')
            return 'authoritative tokens or namespace differ from binding'
        if (recv_req.extra_key is not None or recv_req.lora_id is not None or
                recv_req.positional_embed_overrides is not None or
                (recv_req.return_logprob and recv_req.logprob_start_len >= 0 and
                 recv_req.logprob_start_len < b.reserved_prefix)):
            self.fail_binding(b, actual, 'request changes the validated prefix or its logprob boundary')
            return 'request changes the validated prefix or its logprob boundary'
        if recv_req.sampling_params.max_new_tokens != b.max_new_tokens:
            self.fail_binding(b, actual, 'generation limit differs from binding')
            return 'generation limit differs from binding'
        if any(t.kind == 'restore' and self.handles[t.handle].salt == b.salt for t in self.transfers.values()):
            self.fail_binding(b, actual, 'namespace has a pending host restore')
            return 'namespace has a pending host restore'
        limit = max(0, len(b.ids) - 1)
        if recv_req.return_logprob and recv_req.logprob_start_len >= 0:
            limit = min(limit, recv_req.logprob_start_len)
        m = self._match(b.ids[:limit], b.salt)
        b.input_ids, b.matched, b.full_hit, b.accepted = actual, m.device_prefix_len, m.full_kv_hit_length, True
        return None

    def _release_bind_lock(self, binding: Binding) -> None:
        self.cache.unlock(binding.bind_lock)
        binding.bind_lock = None

    def mark_aborted(self, abort: Any) -> None:
        for rid, binding in self.bindings.items():
            if (binding.accepted and abort.matches(rid) and
                    (binding.req is None or not binding.req.finished())):
                binding.canceled = True

    def fail_binding(self, b: Binding, actual: list[int], message: str) -> None:
        self._release_bind_lock(b)
        self.bindings.pop(b.command['rid'], None)
        self.completions.append(self._reply(b.command, 'error', error=message,
            details=dict(token_ids=actual, full_kv_hit_length=0, residency='absent')))

    def finish(self, req: Any) -> None:
        b = self.bindings.get(req.rid)
        if b is None or not b.accepted or not req.finished():
            return
        from sglang.srt.managers.req_types import FINISH_ABORT
        aborted = b.canceled or isinstance(req.finished_reason, FINISH_ABORT)
        m = self._match(b.ids, b.salt)
        handle = None
        if not aborted:
            handle = uuid.uuid4().hex
            node = m.last_device_node
            h = Handle(b.command['version_id'], b.ids, b.salt, node)
            if b.hold and m.device_prefix_len == len(b.ids):
                h.device_lock = self.cache.lock(node)
                h.leases = 1
            self.handles[handle] = h
        status = ('canceled' if b.canceled else 'error' if aborted else
                  'ready' if m.device_prefix_len == len(b.ids) else 'partial')
        self._release_bind_lock(b)
        self.bindings.pop(req.rid, None)
        self.completions.append(self._reply(b.command, status, handle=handle, matched=b.matched,
            error=req.finished_reason.message if aborted and not b.canceled else None,
            details=dict(token_ids=b.input_ids, full_kv_hit_length=b.full_hit,
                         residency='hbm' if m.device_prefix_len == len(b.ids) else ('host' if m.host_hit_length else 'absent'),
                         checkpoint_tokens=m.device_prefix_len)))
