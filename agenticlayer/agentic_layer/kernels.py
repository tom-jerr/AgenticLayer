"""Page-one HiCache copies, executed inside SGLang's existing transfer fences.

Counters describe submitted bytes, not completed transfers.  The scheduler's
native write/load acknowledgements remain the only completion authority.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import weakref

import torch
import triton
import triton.language as tl


_BLOCK_BYTES = 256
_lock = threading.Lock()
_counters: dict[str, int] = {}
_registered = False
_mamba_pools: weakref.WeakValueDictionary[int, object] = weakref.WeakValueDictionary()


@triton.jit
def _copy_rows_kernel(SRC, DST, SRC_STRIDE, DST_STRIDE, LENGTHS, SRC_IDX, DST_IDX,
                      BLOCK_BYTES: tl.constexpr):
    layer = tl.program_id(0)
    row = tl.program_id(1)
    byte = tl.program_id(2) * BLOCK_BYTES + tl.arange(0, BLOCK_BYTES)
    source = tl.load(SRC + layer).to(tl.pointer_type(tl.uint8))
    target = tl.load(DST + layer).to(tl.pointer_type(tl.uint8))
    src_stride = tl.load(SRC_STRIDE + layer)
    dst_stride = tl.load(DST_STRIDE + layer)
    length = tl.load(LENGTHS + layer)
    src_index = tl.load(SRC_IDX + row).to(tl.int64)
    dst_index = tl.load(DST_IDX + row).to(tl.int64)
    value = tl.load(source + src_index * src_stride + byte, mask=byte < length, other=0)
    tl.store(target + dst_index * dst_stride + byte, value, mask=byte < length)


@dataclass(frozen=True)
class RowLayout:
    """Device tables and validated byte geometry for one multi-layer copy."""
    src_strides: torch.Tensor
    dst_strides: torch.Tensor
    lengths: torch.Tensor
    row_bytes: tuple[int, ...]
    src_capacity: int
    dst_capacity: int
    src_ranges: tuple[tuple[int, int], ...]
    dst_ranges: tuple[tuple[int, int], ...]
    device: torch.device


def _count(name: str, amount: int) -> None:
    if amount:
        with _lock:
            _counters[name] = _counters.get(name, 0) + amount


def transfer_counters() -> dict[str, int]:
    """Return cumulative submitted bytes; keys are stable even before use."""
    names = ("full_triton_bytes", "full_native_bytes", "mamba_triton_bytes",
             "mamba_native_bytes", "mamba_side_bytes")
    with _lock:
        return {f"{direction}_{name}": _counters.get(f"{direction}_{name}", 0)
                for direction in ("h2d", "d2h") for name in names}


_COPY_MODE = os.environ.get("AGENTIC_TRITON_COPY", "0")
if _COPY_MODE not in ("0", "1", "d2h"):
    raise ValueError("AGENTIC_TRITON_COPY must be 0, 1, or d2h")
_TRITON_D2H = _COPY_MODE != "0"
_TRITON_H2D = _COPY_MODE == "1"


def _row_geometry(tensor: torch.Tensor) -> tuple[int, int, int]:
    if tensor.ndim < 1 or tensor.shape[0] == 0:
        raise ValueError("transfer tensors need a nonempty slot dimension")
    expected = 1
    for size, stride in zip(reversed(tensor.shape[1:]), reversed(tensor.stride()[1:])):
        if size > 1 and stride != expected:
            raise ValueError("transfer row is not contiguous")
        expected *= size
    length = expected * tensor.element_size()
    stride = tensor.stride(0) * tensor.element_size()
    if length <= 0 or stride < length:
        raise ValueError("empty or overlapping transfer rows")
    return length, stride, tensor.shape[0]


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError("Agentic Triton copy requires a CUDA device")
    return device.index if device.index is not None else torch.cuda.current_device()


def _mapped(tensor: torch.Tensor, device: torch.device) -> int:
    if tensor.device.type != "cpu":
        raise ValueError("HiCache host tensor must be on CPU")
    # This API calls cudaHostGetDevicePointer; do not use tensor.data_ptr() as a
    # substitute, including on systems where UVA often makes the values equal.
    from sgl_kernel.kvcacheio import get_device_accessible_ptr
    result = int(get_device_accessible_ptr(tensor, _device_index(device)))
    if result == 0:
        raise RuntimeError("host allocation is not mapped for the CUDA device")
    return result


def _descriptor(src: list[tuple[int, torch.Tensor]], dst: list[tuple[int, torch.Tensor]],
                device: torch.device) -> tuple[torch.Tensor, torch.Tensor, RowLayout]:
    if not src or len(src) != len(dst):
        raise ValueError("copy descriptor must have matching nonempty layers")
    src_geo = [_row_geometry(t) for _, t in src]
    dst_geo = [_row_geometry(t) for _, t in dst]
    if any(sg[0] != dg[0] or st.dtype != dt.dtype
           for sg, dg, (_, st), (_, dt) in zip(src_geo, dst_geo, src, dst)):
        raise ValueError("source and destination row geometry differs")
    src_ranges = tuple((p, p + (g[2] - 1) * g[1] + g[0]) for (p, _), g in zip(src, src_geo))
    dst_ranges = tuple((p, p + (g[2] - 1) * g[1] + g[0]) for (p, _), g in zip(dst, dst_geo))
    if any(a < d and c < b for a, b in src_ranges for c, d in dst_ranges):
        raise ValueError("source and destination allocations overlap")
    def table(values):
        return torch.tensor(values, dtype=torch.int64, device=device)
    layout = RowLayout(table([g[1] for g in src_geo]), table([g[1] for g in dst_geo]),
                       table([g[0] for g in src_geo]), tuple(g[0] for g in src_geo),
                       min(g[2] for g in src_geo), min(g[2] for g in dst_geo),
                       src_ranges, dst_ranges, device)
    return table([p for p, _ in src]), table([p for p, _ in dst]), layout


def _indices(src_slots: torch.Tensor, dst_slots: torch.Tensor, device: torch.device,
             src_capacity: int, dst_capacity: int) -> tuple[torch.Tensor, torch.Tensor]:
    if src_slots.ndim != 1 or dst_slots.ndim != 1 or src_slots.numel() != dst_slots.numel():
        raise ValueError("slot indices must be equally sized vectors")
    if src_slots.dtype not in (torch.int32, torch.int64) or dst_slots.dtype not in (torch.int32, torch.int64):
        raise ValueError("slot indices must have integer dtype")
    src = src_slots.to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
    dst = dst_slots.to(device=device, dtype=torch.int64, non_blocking=True).contiguous()
    if src.numel():
        # Reject rather than race on duplicate destinations.  The synchronization
        # is intentional: an asynchronous GPU assertion could arrive after ACK.
        if (int(src.min()) < 0 or int(src.max()) >= src_capacity or
                int(dst.min()) < 0 or int(dst.max()) >= dst_capacity):
            raise ValueError("slot index outside transfer allocation")
        if torch.unique(dst).numel() != dst.numel():
            raise ValueError("duplicate destination slots")
    return src, dst


def copy_rows(src_ptrs: torch.Tensor, dst_ptrs: torch.Tensor, src_slots: torch.Tensor,
              dst_slots: torch.Tensor, layout: RowLayout, *, stream) -> None:
    """Copy arbitrary sparse rows on *stream*; repeated sources are allowed.

    Pointer tables contain CUDA-accessible addresses, including addresses mapped
    at pool construction.  Neither this function nor its callers create fences.
    """
    if stream.device != layout.device:
        raise ValueError("transfer stream is on the wrong device")
    if src_ptrs.device != layout.device or dst_ptrs.device != layout.device:
        raise ValueError("pointer tables are on the wrong device")
    if src_ptrs.numel() != len(layout.row_bytes) or dst_ptrs.numel() != len(layout.row_bytes):
        raise ValueError("pointer table length mismatch")
    with torch.cuda.stream(stream):
        src, dst = _indices(src_slots, dst_slots, layout.device, layout.src_capacity, layout.dst_capacity)
        if not src.numel():
            return
        _copy_rows_kernel[(len(layout.row_bytes), src.numel(), triton.cdiv(max(layout.row_bytes), _BLOCK_BYTES))](
            src_ptrs, dst_ptrs, layout.src_strides, layout.dst_strides, layout.lengths,
            src, dst, BLOCK_BYTES=_BLOCK_BYTES)
        for tensor in (src, dst, src_ptrs, dst_ptrs, layout.src_strides, layout.dst_strides, layout.lengths):
            tensor.record_stream(stream)


def _run(desc, src_indices, dst_indices):
    src, dst, layout = desc
    copy_rows(src, dst, src_indices, dst_indices, layout,
              stream=torch.cuda.current_stream(layout.device))
    return len(src_indices) * sum(layout.row_bytes)


def _mha_init(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if not _TRITON_D2H:
        return result
    if type(self).__name__ != "MHATokenToKVPoolHost":
        raise ValueError("Triton copy does not support specialized MHA host pools")
    if self.page_size != 1 or self.layout != "page_first" or not self.pin_memory:
        raise ValueError("Triton MHA requires pinned page_first, page_size=1")
    device = torch.device("cuda", _device_index(torch.device(self._transfer_device)))
    host = [*self.k_data_refs, *self.v_data_refs]
    devices = [*self.device_k_buffers, *self.device_v_buffers]
    if len(host) != 2 * self.layer_num or len(devices) != len(host):
        raise ValueError("unexpected MHA layer count")
    host_ptrs = [_mapped(t, device) for t in host]
    for t in devices:
        if t.device != device:
            raise ValueError("MHA device buffer on wrong device")
    h = list(zip(host_ptrs, host))
    d = [(t.data_ptr(), t) for t in devices]
    self._agentic_mha_d2h = _descriptor(d, h, device)
    if _TRITON_H2D:
        self._agentic_mha_h2d = [
            _descriptor([h[i], h[i + self.layer_num]], [d[i], d[i + self.layer_num]], device)
            for i in range(self.layer_num)
        ]
    return result


def _mha_backup(original, self, device_pool, host_indices, device_indices, io_backend):
    n = host_indices.numel()
    if _TRITON_D2H and io_backend != "kernel":
        raise ValueError("Triton copy requires kernel HiCache IO backend")
    if _TRITON_D2H and n < 4:
        layout = self._agentic_mha_d2h[2]
        _indices(device_indices, host_indices, layout.device, layout.src_capacity, layout.dst_capacity)
        if not n:
            return None
    if _TRITON_D2H and n >= 4:
        size = _run(self._agentic_mha_d2h, device_indices, host_indices)
        _count("d2h_full_triton_bytes", size)
        return None
    result = original(self, device_pool, host_indices, device_indices, io_backend)
    _count("d2h_full_native_bytes", n * self.get_size_per_token())
    return result


def _mha_load(original, self, device_pool, host_indices, device_indices, layer_id,
              io_backend, *, is_draft=False):
    if not is_draft and not 0 <= layer_id < self.target_layer_num:
        return original(self, device_pool, host_indices, device_indices, layer_id, io_backend, is_draft=is_draft)
    n = host_indices.numel()
    if _TRITON_H2D and io_backend != "kernel":
        raise ValueError("Triton copy requires kernel HiCache IO backend")
    if _TRITON_H2D and n < 4:
        layout = self._agentic_mha_h2d[layer_id][2]
        _indices(host_indices, device_indices, layout.device, layout.src_capacity, layout.dst_capacity)
        if not n:
            return None
    if _TRITON_H2D and n >= 4:
        size = _run(self._agentic_mha_h2d[layer_id], host_indices, device_indices)
        _count("h2d_full_triton_bytes", size)
        return None
    result = original(self, device_pool, host_indices, device_indices, layer_id, io_backend, is_draft=is_draft)
    _count("h2d_full_native_bytes", n * (self.get_size_per_token() // self.layer_num))
    return result


def _mamba_init(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if not _TRITON_D2H:
        return result
    if self.page_size != 1 or self.layout != "page_first" or not self.pin_memory:
        raise ValueError("Triton Mamba requires pinned page_first, page_size=1")
    device = torch.device("cuda", _device_index(torch.device(self._buffer_device)))
    descriptors = {}
    pairs = [(self.temporal_buffer, self.temporal_device_layers)]
    pairs.extend(zip(self.conv_buffer, self.conv_device_layers))
    for host, layers in pairs:
        if not host.numel():
            continue
        if len(layers) != self.num_mamba_layers or any(t.device != device for t in layers):
            raise ValueError("unexpected Mamba layer buffers")
        base = _mapped(host, device)
        views = [host.select(1, i) for i in range(self.num_mamba_layers)]
        h = [(base + i * host.stride(1) * host.element_size(), view) for i, view in enumerate(views)]
        d = [(layer.data_ptr(), layer) for layer in layers]
        descriptors[id(host)] = (
            _descriptor(d, h, device),
            [_descriptor([h[i]], [d[i]], device) for i in range(len(layers))] if _TRITON_H2D else [],
            tuple(t.data_ptr() for t in layers), host,
        )
    self._agentic_mamba_descriptors = descriptors
    for key in descriptors:
        _mamba_pools[key] = self
    # Side state remains in the native method, but resolve it too: an enabled
    # pool with any unmapped host allocation must fail during initialization.
    for host in self.slot_state_buffers:
        if host.numel():
            _mapped(host, device)
    return result


def _mamba_entry(host):
    pool = _mamba_pools.get(id(host))
    if pool is None:
        raise RuntimeError("Mamba host buffer lacks initialized mapped descriptors")
    entry = pool._agentic_mamba_descriptors[id(host)]
    if entry[3] is not host:
        raise RuntimeError("Mamba host buffer identity changed")
    return entry


def _mamba_indices(self, host_indices, device_indices, *, to_device):
    descriptors = self._agentic_mamba_descriptors.values()
    layouts = [entry[0][2] for entry in descriptors]
    host_capacity = min([layout.dst_capacity for layout in layouts] +
                        [t.shape[0] for t in self.slot_state_buffers], default=self.size)
    device_capacity = min([layout.src_capacity for layout in layouts] +
                          [t.shape[0] for t in self.slot_state_device_tensors], default=0)
    device = torch.device("cuda", _device_index(torch.device(self._buffer_device)))
    if to_device:
        return _indices(host_indices, device_indices, device, host_capacity, device_capacity)
    dev, host = _indices(device_indices, host_indices, device, device_capacity, host_capacity)
    return host, dev


def _mamba_load(original, self, device_pool, host_indices, device_indices, layer_id,
                io_backend="kernel", *, is_draft=False):
    if _TRITON_H2D:
        if io_backend != "kernel":
            raise ValueError("Triton copy requires kernel HiCache IO backend")
        host_indices, device_indices = _mamba_indices(self, host_indices, device_indices, to_device=True)
        if not host_indices.numel():
            return None
    result = original(self, device_pool, host_indices, device_indices, layer_id,
                      io_backend, is_draft=is_draft)
    if not _TRITON_H2D:
        layers = [self.temporal_device_layers, *self.conv_device_layers]
        size = sum(layer[layer_id][0].numel() * layer[layer_id].element_size()
                   for layer in layers if layer and layer[layer_id].numel())
        _count("h2d_mamba_native_bytes", host_indices.numel() * size)
    return result


def _mamba_backup(original, self, device_pool, host_indices, device_indices, io_backend="kernel"):
    if _TRITON_D2H:
        if io_backend != "kernel":
            raise ValueError("Triton copy requires kernel HiCache IO backend")
        host_indices, device_indices = _mamba_indices(self, host_indices, device_indices, to_device=False)
        if not host_indices.numel():
            return None
    result = original(self, device_pool, host_indices, device_indices, io_backend)
    if not _TRITON_D2H:
        layers = [self.temporal_device_layers, *self.conv_device_layers]
        size = sum(t[0].numel() * t.element_size()
                   for group in layers for t in group if t.numel())
        _count("d2h_mamba_native_bytes", host_indices.numel() * size)
    return result


def _mamba_load_copy(original, src, dst, src_indices, dst_indices, layer_id, num_layers, io_backend):
    if not _TRITON_H2D:
        return original(src, dst, src_indices, dst_indices, layer_id, num_layers, io_backend)
    if io_backend != "kernel":
        raise ValueError("Triton copy requires kernel HiCache IO backend")
    if not src_indices.numel():
        return None
    _, h2d, ptrs, _ = _mamba_entry(src)
    if len(ptrs) != num_layers or not 0 <= layer_id < num_layers or dst.data_ptr() != ptrs[layer_id]:
        raise ValueError("Mamba restore destination does not match pool layout")
    _count("h2d_mamba_triton_bytes", _run(h2d[layer_id], src_indices, dst_indices))
    return None


def _mamba_backup_copy(original, src_layers, dst, src_indices, dst_indices, num_layers,
                       io_backend, src_ptrs, staging=None, can_use_jit=False):
    if not _TRITON_D2H:
        return original(src_layers, dst, src_indices, dst_indices, num_layers, io_backend,
                        src_ptrs, staging=staging, can_use_jit=can_use_jit)
    if io_backend != "kernel":
        raise ValueError("Triton copy requires kernel HiCache IO backend")
    if not src_indices.numel():
        return None
    d2h, _, ptrs, _ = _mamba_entry(dst)
    if len(ptrs) != num_layers or any(src_layers[i].data_ptr() != ptrs[i] for i in range(num_layers)):
        raise ValueError("Mamba backup source does not match pool layout")
    _count("d2h_mamba_triton_bytes", _run(d2h, src_indices, dst_indices))
    return None


def _mamba_side(original, self, host_indices, device_indices, *, to_device):
    result = original(self, host_indices, device_indices, to_device=to_device)
    size = sum(t[0].numel() * t.element_size() for t in self.slot_state_device_tensors)
    _count(("h2d" if to_device else "d2h") + "_mamba_side_bytes", host_indices.numel() * size)
    return result


_TARGETS = {
    "sglang.srt.mem_cache.pool_host.mha.MHATokenToKVPoolHost.__init__": _mha_init,
    "sglang.srt.mem_cache.pool_host.mha.MHATokenToKVPoolHost.backup_from_device_all_layer": _mha_backup,
    "sglang.srt.mem_cache.pool_host.mha.MHATokenToKVPoolHost.load_to_device_per_layer": _mha_load,
    "sglang.srt.mem_cache.pool_host.mamba.MambaPoolHost.__init__": _mamba_init,
    "sglang.srt.mem_cache.pool_host.mamba.MambaPoolHost.load_to_device_per_layer": _mamba_load,
    "sglang.srt.mem_cache.pool_host.mamba.MambaPoolHost.backup_from_device_all_layer": _mamba_backup,
    "sglang.srt.mem_cache.pool_host.mamba.MambaPoolHost._copy_tensor_pf_lf": _mamba_load_copy,
    "sglang.srt.mem_cache.pool_host.mamba.MambaPoolHost._copy_tensor_all_layers_lf_pf": _mamba_backup_copy,
    "sglang.srt.mem_cache.pool_host.mamba.MambaPoolHost._transfer_slot_states": _mamba_side,
}


def register_transfer_hooks() -> None:
    """Register before SGLang applies plugin hooks; safe to call repeatedly."""
    global _registered
    if _registered:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType
    for target, callback in _TARGETS.items():
        HookRegistry.register(target, callback, HookType.AROUND)
    _registered = True


def assert_transfer_hooks_applied() -> None:
    """Fail closed: the upstream hook registry logs and suppresses patch errors."""
    if not _TRITON_D2H:
        return
    from sglang.srt.plugins.hook_registry import HookRegistry
    missing = set(_TARGETS) - HookRegistry._patched
    if missing:
        raise RuntimeError("Triton transfer hooks not applied: " + ", ".join(sorted(missing)))
