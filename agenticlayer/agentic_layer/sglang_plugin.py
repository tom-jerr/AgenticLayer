"""SGLang entry point and scheduler-owned local control channel."""
from __future__ import annotations

import atexit
import logging
import os
import queue
import socket
import threading
from dataclasses import dataclass, field
from typing import Any

from .protocol import MAX_LINE, WireError, decode_line, encode_line, parse_engine_request

log = logging.getLogger(__name__)
_RETIRE_ABORT = 'sglang.srt.managers.scheduler_components.request_retirer.RequestRetirer.abort'


@dataclass
class _Request:
    envelope: dict[str, Any]
    done: threading.Event = field(default_factory=threading.Event)
    response: dict[str, Any] | None = None


class _ControlSocket:
    def __init__(self, path: str):
        if not os.path.isabs(path):
            raise RuntimeError('AGENTIC_ENGINE_SOCKET must be an absolute path')
        self.queue: queue.Queue[_Request] = queue.Queue(maxsize=256)
        self.pending: _Request | None = None
        self.clients = threading.BoundedSemaphore(64)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Never unlink an existing file, including a stale socket: it may be
        # another experiment's control channel.
        old_mask = os.umask(0o177)
        try:
            self.sock.bind(path)
        finally:
            os.umask(old_mask)
        self.path = path
        self.identity = os.lstat(path).st_ino
        atexit.register(self.close)
        self.sock.listen(64)
        threading.Thread(target=self._listen, name='agentic-control', daemon=True).start()

    def close(self) -> None:
        self.sock.close()
        try:
            if os.lstat(self.path).st_ino == self.identity:
                os.unlink(self.path)
        except FileNotFoundError:
            pass

    def _listen(self) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            if not self.clients.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        try:
            with conn:
                conn.settimeout(30)
                stream = conn.makefile('rb')
                while True:
                    line = stream.readline(MAX_LINE + 1)
                    if not line:
                        break
                    try:
                        envelope = parse_engine_request(decode_line(line))
                        item = _Request(envelope)
                        self.queue.put_nowait(item)
                        # Once accepted, do not time out and later execute an
                        # unacknowledged side effect on the scheduler thread.
                        item.done.wait()
                        response = item.response
                    except (WireError, queue.Full) as exc:
                        response = {'epoch': None, 'replies': [], 'error': str(exc)}
                    conn.sendall(encode_line(response))
        except (OSError, ValueError):
            pass
        finally:
            self.clients.release()

    def drain(self, engine: Any) -> None:
        # Work items contain at most 64 operations. Limit total operations per
        # scheduler pass, retaining ordering within each envelope.
        processed = 0
        while processed < 64:
            if self.pending is not None:
                item, self.pending = self.pending, None
            else:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
            if processed + max(1, len(item.envelope['ops'])) > 64:
                self.pending = item
                break
            try:
                request = item.envelope
                ops = request['ops']
                if request['epoch'] not in (None, engine.epoch):
                    replies = [engine._reply(op, 'stale', error='engine epoch changed') for op in ops]
                elif not ops:
                    replies = [engine.snapshot({})]
                else:
                    replies = []
                    for op in ops:
                        try:
                            replies.append(engine.execute(op))
                        except Exception as exc:
                            log.exception('Agentic engine operation failed')
                            replies.append(engine._reply(op, 'error', error=str(exc)))
                item.response = {'epoch': engine.epoch, 'replies': replies}
                processed += max(1, len(ops))
            finally:
                item.done.set()


def _after_init(original, scheduler, *args, **kwargs):
    result = original(scheduler, *args, **kwargs)
    path = os.environ.get('AGENTIC_ENGINE_SOCKET')
    if not path:
        raise RuntimeError('AGENTIC_ENGINE_SOCKET is required by the AgenticLayer plugin')
    scheduler._agentic_socket = _ControlSocket(path)
    scheduler._agentic_engine = None
    return result


def _after_events(original, scheduler, *args, **kwargs):
    cache = scheduler.tree_cache
    controller = cache.cache_controller
    before_write = list(controller.ack_write_queue) if controller else []
    before_load = list(controller.ack_load_queue) if controller else []
    result = original(scheduler, *args, **kwargs)
    from .engine_cache import EngineCache
    if scheduler._agentic_engine is None:
        scheduler._agentic_engine = EngineCache(scheduler)
    engine = scheduler._agentic_engine
    if controller:
        writes = {id(ack) for ack in controller.ack_write_queue}
        loads = {id(ack) for ack in controller.ack_load_queue}
        engine.observe_acks('offload', [ack for ack in before_write if id(ack) not in writes])
        engine.observe_acks('restore', [ack for ack in before_load if id(ack) not in loads])
    engine.poll_transfers()
    scheduler._agentic_socket.drain(engine)
    return result


def _before_generate(original, scheduler, recv_req, *args, **kwargs):
    engine = scheduler._agentic_engine
    # The scheduler normally drains at the next-batch boundary; a request can
    # arrive before its first boundary following startup.
    if engine is None:
        from .engine_cache import EngineCache
        engine = scheduler._agentic_engine = EngineCache(scheduler)
        scheduler._agentic_socket.drain(engine)
    error = engine.accept(recv_req)
    if error is None:
        result = original(scheduler, recv_req, *args, **kwargs)
        binding = engine.bindings.get(recv_req.rid)
        if binding is not None and binding.req is not None and binding.req.finished():
            engine.finish(binding.req)
        return result
    from sglang.srt.managers.io_struct import AbortReq
    from http import HTTPStatus
    abort = AbortReq(rid=recv_req.rid, abort_message=error,
                     finished_reason={'type': 'abort', 'status_code': HTTPStatus.BAD_REQUEST, 'message': error})
    scheduler.send_control_output(abort, recv_req)


def _after_batch(original, scheduler, batch, result, *args, **kwargs):
    reqs = tuple(batch.reqs)
    value = original(scheduler, batch, result, *args, **kwargs)
    engine = scheduler._agentic_engine
    if engine is not None:
        for req in reqs:
            engine.finish(req)
    return value


def _around_abort(original, scheduler, recv_req, *args, **kwargs):
    engine = scheduler._agentic_engine
    if engine is not None:
        engine.mark_aborted(recv_req)
    return original(scheduler, recv_req, *args, **kwargs)


def _after_retire_abort(original, retirer, req, *args, **kwargs):
    result = original(retirer, req, *args, **kwargs)
    # Queued or mid-prefill aborts may retire outside process_batch_result.
    # Publish only after native retirement has released the request's memory.
    engine = getattr(retirer.tree_cache, '_agentic_engine', None)
    if engine is not None:
        engine.finish(req)
    return result


def _after_create_req(original, scheduler, recv_req, *args, **kwargs):
    req = original(scheduler, recv_req, *args, **kwargs)
    engine = scheduler._agentic_engine
    if engine is not None:
        binding = engine.bindings.get(recv_req.rid)
        if binding is not None and binding.accepted:
            binding.req = req
            req._agentic_binding = binding
    return req


def _after_match(original, req, *args, **kwargs):
    result = original(req, *args, **kwargs)
    binding = getattr(req, '_agentic_binding', None)
    if binding is not None and not binding.match_observed:
        binding.matched = result.device_prefix_len
        binding.full_hit = result.full_kv_hit_length
        binding.match_observed = True
    return result


def build_cache(ctx):
    from sglang.srt.mem_cache.registry import create_unified_radix_cache
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.mem_cache.unified_cache.components import ComponentType
    from sglang.srt.plugins.hook_registry import HookRegistry
    from sglang.srt.runtime_context import get_device
    from .kernels import assert_transfer_hooks_applied

    if os.environ.get('SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND') != 'python':
        raise RuntimeError('AgenticLayer requires the Python TreeCore')
    if ctx.tp_size != 1 or ctx.params.page_size != 1 or not ctx.is_hybrid_ssm or ctx.disable_radix_cache:
        raise RuntimeError('AgenticLayer requires TP=1, page_size=1 and enabled hybrid SSM radix cache')
    if get_device().sleep_on_idle:
        raise RuntimeError('AgenticLayer requires --sleep-on-idle disabled so the scheduler drains control requests')
    assert_transfer_hooks_applied()
    prefix = 'sglang.srt.managers.scheduler.Scheduler.'
    required = ('init_ipc_channels', '_process_hicache_events', 'handle_generate_request',
                'process_batch_result', 'abort_request', '_create_generation_req')
    if 'sglang.srt.managers.schedule_batch.match_kv_cache' not in HookRegistry._patched:
        raise RuntimeError('AgenticLayer native match hook not installed')
    if _RETIRE_ABORT not in HookRegistry._patched:
        raise RuntimeError('AgenticLayer native abort retirement hook not installed')
    missing = [name for name in required if prefix + name not in HookRegistry._patched]
    if missing:
        raise RuntimeError('AgenticLayer scheduler hooks not installed: ' + ', '.join(missing))

    class AgenticUnifiedRadixCache(UnifiedRadixCache):
        """Native hybrid cache; control state lives in the scheduler owner."""

    cache = create_unified_radix_cache(ctx, cache_class=AgenticUnifiedRadixCache)
    if cache._tree_core_backend != 'python' or not {ComponentType.FULL, ComponentType.MAMBA}.issubset(cache.components):
        raise RuntimeError('AgenticLayer requires native Full and Mamba components on Python TreeCore')
    return cache


def register() -> None:
    from sglang.srt.mem_cache.registry import register_radix_cache_backend
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    from .kernels import register_transfer_hooks

    register_transfer_hooks()
    register_radix_cache_backend('agentic_layer', build_cache)
    HookRegistry.register('sglang.srt.managers.schedule_batch.match_kv_cache', _after_match, HookType.AROUND)
    HookRegistry.register(_RETIRE_ABORT, _after_retire_abort, HookType.AROUND)
    prefix = 'sglang.srt.managers.scheduler.Scheduler.'
    for name, hook in (('init_ipc_channels', _after_init),
                       ('_process_hicache_events', _after_events),
                       ('handle_generate_request', _before_generate),
                       ('process_batch_result', _after_batch),
                       ('abort_request', _around_abort),
                       ('_create_generation_req', _after_create_req)):
        HookRegistry.register(prefix + name, hook, HookType.AROUND)
