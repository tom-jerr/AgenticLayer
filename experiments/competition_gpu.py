"""Schedule the same two real GPU consumers at a confirmed shared barrier."""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from pathlib import Path

import httpx

from .gpu_run import ENGINE, engine, write_json
from .hybrid_gpu import INGRESS, ingress
from .prepare_gpu import WAIT, Branch, action, chat, reset_cache, target_text


class Decisions:
    def __init__(self, path):
        self.log = path.open(encoding="utf-8")
        self.actions = {}

    def read(self):
        for line in self.log:
            if "agentic_layer:policy " in line:
                row = json.loads(line.split("agentic_layer:policy ", 1)[1])
                self.actions.setdefault(row["action_id"], []).append(row)

    async def competing(self, *ids):
        async with asyncio.timeout(2):
            while not all(i in self.actions for i in ids):
                self.read()
                await asyncio.sleep(.005)
        assert all(self.actions[i][0]["reason"] == "illegal" for i in ids), ids

    def close(self):
        self.log.close()


async def generation(b, body):
    work = action("model")
    permit = (await b.event("ready", work))["permit"]
    assert permit["version_id"] == b.version and permit["expires_mono_ns"] > time.monotonic_ns()
    data = await b.http.post("/v1/chat/completions", json=body, headers={"X-Agentic-Session": b.id,
        "X-Agentic-Branch": b.id, "X-Agentic-Version": b.version, "X-Agentic-Request": work["rid"]})
    completed = time.monotonic_ns()
    data.raise_for_status()
    await b.event("commit", {"action_id": work["action_id"]})
    return data.json(), work["rid"], completed


async def run_pair(http, log, decisions, wait_ms, child_service_ms):
    run = f"competition-{uuid.uuid4().hex}"
    front, child, gate = (Branch(http, run) for _ in range(3))
    ids = [uuid.uuid4().hex], [uuid.uuid4().hex], [uuid.uuid4().hex], [uuid.uuid4().hex]
    for branch, messages in zip((front, child, gate), ids[:3], strict=True):
        await branch.event("append", {"isolation_domain": branch.id, "message_ids": messages})
    barrier = action("tool")
    await gate.event("ready", barrier)
    await gate.event("start", {"action_id": barrier["action_id"]})
    dep = [barrier["action_id"]]
    body = chat(target_text())
    warm = action("prepare", chat=body, wait_type=WAIT, dependencies=dep,
                  boundary_message_ids=ids[0], target_message_ids=ids[3])
    warm_ack = await front.event("ready", warm)
    launch = action("child", dependencies=dep, service_ms=child_service_ms)

    async def child_lane():
        await child.event("ready", launch)
        await child.event("start", {"action_id": launch["action_id"]})
        # A separate branch performs the child's actual model work.
        worker = Branch(http, run)
        await worker.event("append", {"isolation_domain": worker.id, "message_ids": [uuid.uuid4().hex]})
        result, rid, done = await generation(worker, chat("Answer exactly 7 with no other text."))
        await worker.event("end", {})
        await child.event("finish", {"action_id": launch["action_id"], "status": "completed"})
        await child.event("commit", {"action_id": launch["action_id"]})
        await child.event("end", {})
        return result, rid, done

    async with asyncio.TaskGroup() as group:
        running = group.create_task(child_lane())
        await decisions.competing(warm["action_id"], launch["action_id"])
        await gate.event("finish", {"action_id": barrier["action_id"], "status": "completed"})
        open_ns = time.monotonic_ns()
        await gate.event("commit", {"action_id": barrier["action_id"]})
        await asyncio.sleep(max(0., wait_ms / 1000 - (time.monotonic_ns() - open_ns) / 1e9))
        prepared = (await front.lookup(warm_ack["event_id"]))["status"] == "completed"
        boundary_ns = time.monotonic_ns()
        change = {"message_ids": ids[3]}
        if prepared:
            change["candidate_action_id"] = warm["action_id"]
        await front.event("replace", change)
        response, rid, done = await generation(front, body)
    child_response, child_rid, child_done = running.result()
    await front.event("end", {})
    await gate.event("end", {})
    assert response["choices"][0]["message"]["content"] == "42", response
    assert child_response["choices"][0]["message"]["content"] == "7", child_response
    decisions.read()
    events = [d for a in (warm, launch) for d in decisions.actions[a["action_id"]]]
    selected = sorted((d for d in events if d["reason"] == "selected"), key=lambda d: d["mono_ns"])
    assert selected, events
    first = selected[0]
    same_decision = [row for row in events if row["mono_ns"] == first["mono_ns"]]
    assert len(same_decision) == 2 and any(row["reason"] == "not_selected" for row in same_decision), same_decision
    sample = {"run_id": run, "first": first["kind"], "prepared": prepared, "foreground_rid": rid, "child_rid": child_rid,
              "foreground_ms": (done-boundary_ns)/1e6, "child_ms": (child_done-open_ns)/1e6,
              "makespan_ms": (max(done, child_done)-open_ns)/1e6, "decisions": events}
    log.write(json.dumps(sample)+"\n")
    return sample


async def measure(root, name, token, count, wait_ms):
    with (root / "logs" / f"{name}-pairs.jsonl").open("x", buffering=1) as log:
        async with httpx.AsyncClient(base_url=INGRESS, headers={"Authorization": f"Bearer {token}"},
                                     timeout=180, trust_env=False) as http:
            # Get the ranking estimate from this model and runtime, not a mock child.
            child_times = []
            for i in range(10):
                await reset_cache(http)
                begin = time.monotonic_ns()
                res = await http.post(ENGINE+"/v1/chat/completions", json={**chat("Answer exactly 7 with no other text."),
                    "cache_salt": uuid.uuid4().hex})
                end = time.monotonic_ns()
                res.raise_for_status()
                assert res.json()["choices"][0]["message"]["content"] == "7", res.json()
                if i >= 5:
                    child_times.append((end-begin)/1e6)
            service = statistics.median(child_times)
            rows = []
            decisions = Decisions(root / "logs" / f"{name}-ingress.log")
            try:
                for _ in range(count):
                    await reset_cache(http)
                    rows.append(await run_pair(http, log, decisions, wait_ms, service))
            finally:
                decisions.close()
    return {"scope": "injected shared legal barrier: admit child vs prepare; not E5 three-way or trace", "child_samples_ms": child_times,
            "first": [r["first"] for r in rows], "runs": rows,
            "p50_ms": {k: statistics.median(r[k] for r in rows) for k in ("foreground_ms", "child_ms", "makespan_ms")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--wait-ms", type=int, default=800)
    a = p.parse_args()
    for mode in ("strong", "joint"):
        name = a.name+"-"+mode
        with engine(a.run_dir, "extended", name), ingress(a.run_dir, name, mode, a.calibration) as token:
            result = asyncio.run(measure(a.run_dir, name, token, a.runs, a.wait_ms))
        write_json(a.run_dir / f"{name}.json", result)
        print(mode, {k: result[k] for k in ("first", "p50_ms")}, flush=True)


if __name__ == "__main__":
    main()
