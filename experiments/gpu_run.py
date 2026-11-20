"""Launch one owned pinned engine at a time and record native baseline evidence."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import signal
import socket
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import httpx

PROJECT = Path(__file__).resolve().parents[1]
MODEL = Path("/root/.cache/huggingface/hub/models--AxionML--Qwen3.5-9B-NVFP4/snapshots/97aef92393f126bf649f310cd40861be8dad3279")
ENGINE = "http://127.0.0.1:30080"
ARMS = ("native-radix", "native-hicache", "reference-cold", "extended", "extended-hicache")
LENGTHS = (512, 4096, 16384)
SEEDS = (42, 43, 44)


def write_json(path: Path, data) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def engine_env(root: Path, arm: str) -> dict[str, str]:
    toolkit = root / "env/cuda-toolchain/nvidia/cu13"
    python = root / "env/python/bin/python"
    # SGLang startup uses requests for its own health/model warmup. Do not
    # route that loopback traffic through an inherited outbound proxy.
    bypass = ",".join(filter(None, (os.getenv("NO_PROXY"), os.getenv("no_proxy"), "127.0.0.1", "localhost", "::1")))
    sources = [str(root / "src/sglang/python")]
    if arm.startswith("extended"):
        sources.insert(0, str(PROJECT / "agenticlayer"))
    return {**os.environ, "RUN": str(root), "ENGINE_ARM": arm, "PYTHON": str(python), "MODEL": str(MODEL),
            "PYTHONPATH": os.pathsep.join(sources), "PYTHONDONTWRITEBYTECODE": "1", "CUDA_HOME": str(toolkit),
            "PATH": f"{toolkit}/bin:{python.parent}:{os.environ['PATH']}", "HF_HOME": str(MODEL.parents[3]),
            "SGLANG_CACHE_DIR": str(root / "cache/sglang"), "TRITON_CACHE_DIR": str(root / "cache/triton"),
            "CUDA_CACHE_PATH": str(root / "cache/cuda"), "TMPDIR": str(root / "tmp"),
            "FLASHINFER_WORKSPACE_DIR": str(root / "cache/flashinfer"), "MAX_JOBS": os.getenv("MAX_JOBS", "3"),
            "NO_PROXY": bypass, "no_proxy": bypass}


def port_available(port: int) -> None:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))

GRAPH_BUCKETS = list(range(1, 9))


def _verify_fixed_graph(client: httpx.Client, log_path: Path) -> dict:
    response = client.get("/server_info")
    response.raise_for_status()
    info = response.json()
    config = info["cuda_graph_config"]
    decode = config["decode"]
    if (decode["backend"], decode["max_bs"], decode["bs"], config["prefill"]["backend"]) != (
            "full", GRAPH_BUCKETS[-1], GRAPH_BUCKETS, "disabled"):
        raise RuntimeError(f"mismatched CUDA graph settings: {config}")
    log = log_path.read_text()
    capture = re.findall(r"Capture target decode CUDA graph end\. bs=(\[[^\n]+?\])", log)
    if len(capture) != 1 or json.loads(capture[0]) != GRAPH_BUCKETS or "Disable prefill CUDA graph" not in log:
        raise RuntimeError(f"missing actual fixed-bucket decode capture or prefill-disabled log: {log_path}")
    graph = {"cuda_graph_config": config, "captured_decode_buckets": json.loads(capture[0]),
             "max_running_requests": info["max_running_requests"],
             "hybrid_cache_checkpoints_level": info["hybrid_cache_checkpoints_level"]}
    if os.getenv("AGENTIC_DETERMINISTIC") == "1":
        if info.get("enable_deterministic_inference") is not True:
            raise RuntimeError(f"deterministic inference was requested but not enabled: {log_path}")
        graph["enable_deterministic_inference"] = True
    return graph



@contextmanager
def engine(root: Path, arm: str, name: str):
    port_available(30080)
    command = ["bash", str(PROJECT / "experiments/launch_engine.sh")]
    env = engine_env(root, arm)
    if env.get("AGENTIC_FIXED_GRAPH") == "1" and env.get("AGENTIC_DECODE_GRAPH") != "full":
        raise RuntimeError("fixed graph experiments require AGENTIC_DECODE_GRAPH=full")
    path = root / "logs" / f"{name}.log"
    socket_path = root / "engine.sock" if arm.startswith("extended") else None
    if socket_path is not None and (socket_path.exists() or socket_path.is_symlink()):
        raise RuntimeError("engine.sock already exists; refusing to unlink an unknown socket")
    with path.open("xb") as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        socket_info = None
        try:
            with httpx.Client(base_url=ENGINE, trust_env=False, timeout=180) as client:
                for _ in range(450):
                    if socket_info is None and socket_path is not None and socket_path.is_socket():
                        socket_info = socket_path.stat()
                    if child.poll() is not None:
                        raise RuntimeError(f"{arm} exited {child.returncode}; see {path}")
                    try:
                        if client.get("/health", timeout=2).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(2)
                else:
                    raise TimeoutError(f"{arm} startup, see {path}")
                if env.get("AGENTIC_FIXED_GRAPH") == "1":
                    graph = _verify_fixed_graph(client, path)
                    write_json(root / "logs" / f"{name}-graph.json", graph)
                yield client
        finally:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            if socket_info:
                if socket_path.is_socket() and socket_path.stat().st_ino == socket_info.st_ino:
                    socket_path.unlink()


class Probe:
    def __init__(self, client: httpx.Client, path: Path):
        self.client = client
        self.log = path.open("x", encoding="utf-8", buffering=1)
        self.salt = uuid.uuid4().hex

    def close(self) -> None:
        self.log.close()

    def get(self, endpoint: str):
        response = self.client.get(endpoint)
        self.log.write(json.dumps({"endpoint": endpoint, "status": response.status_code, "body": response.text}) + "\n")
        response.raise_for_status()
        return response

    def gen(self, tag: str, ids: list[int], count: int, *, salt: str | None = None, logprob: int | None = None):
        request = {"input_ids": ids, "cache_salt": self.salt + (salt or uuid.uuid4().hex),
                   "sampling_params": {"max_new_tokens": count, "ignore_eos": True, "temperature": 0}}
        if logprob is not None:
            request.update(return_logprob=True, logprob_start_len=logprob, top_logprobs_num=5)
        start = time.monotonic_ns()
        res = self.client.post("/generate", json=request)
        elapsed = (time.monotonic_ns() - start) / 1e6
        try:
            data = res.json()
        except ValueError:
            data = {"body": res.text}
        self.log.write(json.dumps({"tag": tag, "wall_ms": elapsed, "request": request, "status": res.status_code,
                                   "response": data}, separators=(",", ":")) + "\n")
        res.raise_for_status()
        assert data["meta_info"]["prompt_tokens"] == len(ids), (tag, data["meta_info"])
        assert len(data["output_ids"]) == count, (tag, data["meta_info"])
        return data, elapsed


def token_fixture(n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    return [rng.randrange(1000, 17000) for _ in range(n)]


def configuration(probe: Probe, arm: str):
    probe.get("/health_generate")
    info = probe.get("/server_info").json()
    models = probe.get("/v1/models").json()
    flags = {"served_model_name": "qwen35-9b-nvfp4", "page_size": 1, "tp_size": 1, "dtype": "bfloat16",
             "kv_cache_dtype": "bf16", "attention_backend": "triton", "linear_attn_prefill_backend": "triton",
             "linear_attn_decode_backend": "triton", "quantization": "modelopt_fp4",
             "fp4_gemm_runner_backend": "flashinfer_cutlass", "context_length": 32774, "max_req_input_len": 32768,
             "disable_radix_cache": arm == "reference-cold", "enable_hierarchical_cache": arm == "native-hicache"}
    for key, expected in flags.items():
        assert info.get(key) == expected, (key, expected, info.get(key))
    assert any(m["id"] == "qwen35-9b-nvfp4" for m in models["data"])
    return flags


def radix_smoke(p: Probe) -> dict:
    ids = token_fixture(32767, 42)
    long, context_ms = p.gen("context-32767", ids, 1)
    cold, cold_ms = p.gen("cold", ids[:1025], 1, logprob=-1)
    p.gen("prime", ids[:1024], 0, salt="warm")
    warm, warm_ms = p.gen("warm", ids[:1025], 1, salt="warm", logprob=-1)
    cm, wm = cold["meta_info"], warm["meta_info"]
    delta = abs(cm["output_token_logprobs"][0][0] - wm["output_token_logprobs"][0][0])
    assert warm["output_ids"] == cold["output_ids"] and wm["cached_tokens"] > cm["cached_tokens"], (cm, wm)
    assert delta <= 1e-4, ("cold/warm logprob", delta)
    return {"context_ms": context_ms, "context_output": long["output_ids"], "cold_ms": cold_ms, "warm_ms": warm_ms,
            "cold_hit": cm["cached_tokens"], "warm_hit": wm["cached_tokens"], "logprob_delta": delta}


def cold_reference(p: Probe) -> dict:
    evidence = {}
    for length in LENGTHS:
        for seed in SEEDS:
            prompt = token_fixture(length, seed)
            runs = [p.gen(f"cold-{length}-{seed}-{i}", prompt, 8, logprob=-1)[0] for i in range(5)]
            target = runs[0]["output_ids"]
            forced_prompt = prompt + target
            # The API returns a null score at logprob_start_len itself.
            # Include the preceding prompt token, so the first forced token
            # is scored by the same extend forward as the rest.
            scores = [p.gen(f"forced-{length}-{seed}-{i}", forced_prompt, 0, logprob=length - 1)[0]["meta_info"]
                      for i in range(5)]
            expected = [[(row[0], row[1]) for row in s["input_token_logprobs"] if row[0] is not None] for s in scores]
            assert all([token for _, token in sample] == target for sample in expected), (length, seed, expected)
            evidence[f"{length}/{seed}"] = {"greedy": [run["output_ids"] for run in runs], "forced_ids": target,
                "forced_logprob": [[score for score, _ in row] for row in expected],
                "forced_top5": [score["input_top_logprobs"][-len(target):] for score in scores]}
    return evidence


def hicache_smoke(p: Probe, log: Path) -> dict:
    text = log.read_text(errors="replace")
    capacity = int(re.findall(r"KV Cache is allocated.*?#tokens: (\d+)", text)[-1])
    ids = token_fixture(16384 + 1, 20261119)
    cold, _ = p.gen("l2-cold-reference", ids, 1, logprob=-1)
    p.gen("l2-prime", ids[:-1], 0, salt="target")
    for i in range(capacity // 32767 + 2):
        p.gen(f"pressure-{i}", token_fixture(32767, i + 330), 1)
    result, wall = p.gen("l2-restore", ids, 1, salt="target", logprob=-1)
    meta = result["meta_info"]
    host = (meta.get("cached_tokens_details") or {}).get("host", 0)
    assert host > 0, ("no native host hit under pressure", meta)
    assert result["output_ids"] == cold["output_ids"], (cold["output_ids"], result["output_ids"])
    return {"host_tokens": host, "full_capacity": capacity, "wall_ms": wall}


def native(root: Path, arm: str, name: str, attach: bool) -> None:
    @contextmanager
    def attached():
        with httpx.Client(base_url=ENGINE, timeout=180, trust_env=False) as client:
            yield client

    with (attached() if attach else engine(root, arm, name)) as client:
        p = Probe(client, root / "logs" / f"{name}-requests.jsonl")
        try:
            flags = configuration(p, arm)
            if arm == "native-radix":
                measurements = radix_smoke(p)
            elif arm == "reference-cold":
                measurements = cold_reference(p)
            else:
                measurements = hicache_smoke(p, root / "logs" / f"{name}.log")
            write_json(root / f"{name}.json", {"arm": arm, "flags": flags, "measurements": measurements})
            print(name, measurements if arm != "reference-cold" else f"{len(measurements)} cold envelopes", flush=True)
        finally:
            p.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS[:3])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name")
    parser.add_argument("--attach", action="store_true", help="Use an already running owned native engine")
    args = parser.parse_args()
    native(args.run_dir.resolve(), args.arm, args.name or args.arm, args.attach)


if __name__ == "__main__":
    main()
