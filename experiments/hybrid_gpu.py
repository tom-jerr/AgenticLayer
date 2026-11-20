"""Exercise the real scheduler socket, Full+Mamba retention and host restoration."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import signal
import subprocess
import time
import uuid
from contextlib import contextmanager, suppress
from pathlib import Path

import httpx

from .gpu_run import ENGINE, MODEL, PROJECT, engine, engine_env, port_available, token_fixture, write_json

INGRESS = "http://127.0.0.1:30081"


def ident() -> str:
    return uuid.uuid4().hex


@contextmanager
def ingress(root: Path, name: str, policy: str, calibration: Path | None = None):
    port_available(30081)
    env = engine_env(root, "extended-hicache")
    token = secrets.token_hex(32)
    tokenizer = hashlib.sha256()
    for path in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        tokenizer.update((MODEL / path).read_bytes())
    env.update(AGENTIC_ENGINE_SOCKET=str(root / "engine.sock"), AGENTIC_LOCAL_TOKEN=token,
               AGENTIC_POLICY=policy, AGENTIC_TOKENIZER_HASH=tokenizer.hexdigest(),
               AGENTIC_LAYOUT_HASH=hashlib.sha256((MODEL / "config.json").read_bytes() + b"tp1/page1/bf16/fp32/no_buffer").hexdigest(),
               PYTHONPATH=f"{PROJECT}/agenticlayer:{env['PYTHONPATH']}")
    if calibration:
        env["AGENTIC_CALIBRATION"] = str(calibration.resolve())
    else:
        env.pop("AGENTIC_CALIBRATION", None)
    args = [env["PYTHON"], "-c", "import logging; logging.basicConfig(level=logging.INFO); from agentic_layer.server import main; main()"]
    with (root / "logs" / f"{name}-ingress.log").open("xb") as log:
        proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            with httpx.Client(base_url=INGRESS, headers={"Authorization": f"Bearer {token}"}, trust_env=False) as h:
                for _ in range(100):
                    if proc.poll() is not None:
                        raise RuntimeError(f"ingress exited {proc.returncode}; see {log.name}")
                    try:
                        if h.get("/health", timeout=1).status_code == 200:
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.1)
                else:
                    raise TimeoutError(f"ingress startup; see {log.name}")
            yield token
        finally:
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


class Driver:
    def __init__(self, control, http: httpx.AsyncClient, log, run: str):
        self.control, self.http, self.log, self.run = control, http, log, run

    def record(self, tag: str, data) -> None:
        self.log.write(json.dumps({"tag": tag, "mono_ns": time.monotonic_ns(), "data": data}, separators=(",", ":")) + "\n")

    def command(self, op: str, *, rid: str | None = None, version: str, handle: str | None = None, payload=None):
        action = ident()
        d = {"op": op, "action_id": action, "run_id": self.run, "rid": rid or action, "version_id": version, "handle": handle}
        if payload is not None:
            d["payload"] = payload
        return d

    async def rpc(self, *commands):
        self.record("command", commands)
        result = await self.control.rpc(list(commands))
        self.record("command-ack", result)
        return result

    async def wait(self, future):
        async with asyncio.timeout(30):
            while not future.done():
                await self.control.poll()
                if not future.done():
                    await asyncio.sleep(0.003)
        result = future.result()
        self.record("completion", result)
        return result

    async def generate(self, ids, salt, version, *, n=0, prepare=False, logprob=None):
        rid = ident()
        bind = self.command("bind", rid=rid, version=version, payload={"expected_ids": ids, "cache_salt": salt,
            "kind": "prepare" if prepare else "batch", "session_id": version, "branch_id": version,
            "max_new_tokens": n, "hold": prepare})
        future = self.control.watch(bind["action_id"])
        try:
            start = time.monotonic_ns()
            reply, = await self.rpc(bind)
            assert reply["status"] == "ok", reply
            body = {"input_ids": ids, "rid": rid, "cache_salt": salt,
                    "sampling_params": {"max_new_tokens": n, "temperature": 0, "ignore_eos": True}}
            if logprob is not None:
                body.update(return_logprob=True, logprob_start_len=logprob, top_logprobs_num=5)
            res = await self.http.post("/generate", json=body)
            self.record("native", {"status": res.status_code, "request": body, "response": res.json()})
            res.raise_for_status()
            finish = await self.wait(future)
            assert finish["details"]["token_ids"] == ids and finish["status"] in ("ready", "partial"), finish
            return res.json(), finish, (time.monotonic_ns() - start) / 1e6
        finally:
            self.control.unwatch(bind["action_id"])

    async def transfer(self, op, version, handle, expected, *, protect=False):
        command = self.command(op, version=version, handle=handle)
        future = self.control.watch(command["action_id"])
        try:
            start = time.monotonic_ns()
            if protect:
                issued, reclaim = await self.rpc(command, self.command("reclaim", version=version, handle=handle))
                assert reclaim["status"] == "unavailable" and "in-flight" in reclaim["error"], (issued, reclaim)
            else:
                issued, = await self.rpc(command)
            assert issued["status"] == "accepted", issued
            reply = await self.wait(future)
            assert reply["status"] == expected, reply
            return reply, (time.monotonic_ns() - start) / 1e6
        finally:
            self.control.unwatch(command["action_id"])


async def verify(root: Path, name: str, copy_mode: str):
    from agentic_layer.engine_client import EngineClient

    control = EngineClient(root / "engine.sock")
    await control.handshake()
    assert all(control.capabilities[x] for x in ("offload", "restore", "reclaim", "prepare"))
    results = []
    with (root / "logs" / f"{name}-wire.jsonl").open("x", encoding="utf-8", buffering=1) as log:
        async with httpx.AsyncClient(base_url=ENGINE, trust_env=False, timeout=180) as http:
            d = Driver(control, http, log, name)
            d.record("handshake", control.snapshot)
            for length in (512, 4096, 16384):
                version, salt = ident(), hashlib.sha256(ident().encode()).hexdigest()
                ids = token_fixture(length + 1, 42)
                _, prepare, prefill_ms = await d.generate(ids[:-1], salt, version, prepare=True)
                assert prepare["status"] == "ready" and prepare["handle"], prepare
                handle = prepare["handle"]
                locked, release = await d.rpc(d.command("reclaim", version=version, handle=handle),
                                              d.command("release", version=version, handle=handle))
                assert locked["status"] == "unavailable" and release["status"] == "ok", (locked, release)
                off, off_ms = await d.transfer("offload", version, handle, "completed", protect=True)
                assert off["details"]["demoted"] and off["details"]["residency"] == "host", off
                on, on_ms = await d.transfer("restore", version, handle, "ready")
                cached, ref, _ = await d.generate(ids, salt, ident(), n=8, logprob=-1)
                fresh, _, _ = await d.generate(ids, hashlib.sha256(ident().encode()).hexdigest(), ident(), n=8, logprob=-1)
                assert ref["matched_tokens"] >= length, ref
                assert cached["output_ids"] == fresh["output_ids"], (cached["output_ids"], fresh["output_ids"])
                ca, fr = (x["meta_info"]["output_token_logprobs"] for x in (cached, fresh))
                max_delta = max(abs(a[0] - b[0]) for a, b in zip(ca, fr, strict=True))
                row = {"length": length, "prefill_ms": prefill_ms, "d2h_wall_ms": off_ms, "h2d_wall_ms": on_ms,
                       "d2h_transfer_us": off["transfer_us"], "h2d_transfer_us": on["transfer_us"],
                       "d2h_bytes": off["transfer_bytes"], "h2d_bytes": on["transfer_bytes"],
                       "ack_metrics_complete": [off["details"]["transfer_metrics_complete"], on["details"]["transfer_metrics_complete"]],
                       "matched_tokens": ref["matched_tokens"], "max_logprob_delta": max_delta}
                d.record("physical-restore", row)
                results.append(row)
                print(name, row, flush=True)
            await control.poll()
            final = control.snapshot
            d.record("final-snapshot", final)
            counters = final["details"]["transfer_counters"]
            for direction in ("d2h", "h2d"):
                triton = copy_mode == "1" or (copy_mode == "d2h" and direction == "d2h")
                kind = "triton" if triton else "native"
                other = "native" if triton else "triton"
                for pool in ("full", "mamba"):
                    assert counters[f"{direction}_{pool}_{kind}_bytes"] > 0, counters
                    assert counters[f"{direction}_{pool}_{other}_bytes"] == 0, counters
    return {"arm": name, "copy_mode": copy_mode, "runs": results, "transfer_counters": counters}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    copies = parser.add_mutually_exclusive_group()
    copies.add_argument("--triton", action="store_const", const="1", dest="copy_mode")
    copies.add_argument("--triton-d2h", action="store_const", const="d2h", dest="copy_mode")
    parser.set_defaults(copy_mode="0")
    parser.add_argument("--attach", action="store_true")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if args.attach:
        result = asyncio.run(verify(root, args.name, args.copy_mode))
    else:
        os.environ["AGENTIC_TRITON_COPY"] = args.copy_mode
        with engine(root, "extended-hicache", args.name):
            result = asyncio.run(verify(root, args.name, args.copy_mode))
    write_json(root / f"{args.name}.json", result)


if __name__ == "__main__":
    main()
