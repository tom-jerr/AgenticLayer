"""Streaming FreeInference profiling and deterministic, causal trace projection.

The dataset is read-only.  Artifacts are written only under the supplied run
folder.  A profile must precede selection and pins every compressed source.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import zstandard

STRATA = ("subsession", "mutation_switch", "multi_call", "ordinary")
VERSION = "agentic-kv-v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".trace-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(data)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _sources(dataset):
    root = Path(dataset).resolve() / "release"
    weeks = sorted(p for p in root.iterdir() if p.is_dir() and (p / "traces").is_dir())
    if len(weeks) != 16:
        raise ValueError(f"expected the locked 16 release weeks, found {len(weeks)}")
    files = []
    for week in weeks:
        paths = sorted((week / "traces").glob("*.jsonl.zst"))
        if not paths:
            raise ValueError(f"week has no trace files: {week.name}")
        files.extend((week.name, p) for p in paths)
    return root, [p.name for p in weeks], files


class _HashReader(io.RawIOBase):
    def __init__(self, raw):
        self.raw = raw
        self.digest = hashlib.sha256()

    def readable(self):
        return True

    def readinto(self, buffer):
        count = self.raw.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


def _sessions(root, files, source_hashes):
    for week, path in files:
        with path.open("rb") as raw:
            reader = _HashReader(raw)
            with zstandard.ZstdDecompressor().stream_reader(reader, closefd=False) as stream:
                with io.BufferedReader(stream) as lines:
                    for line_number, line in enumerate(lines, 1):
                        try:
                            session = json.loads(line)
                        except (ValueError, UnicodeDecodeError) as exc:
                            raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                        if session.get("week") != week:
                            raise ValueError(f"week mismatch at {path}:{line_number}")
                        yield week, session, {"file": str(path.relative_to(root)), "line": line_number}
            source_hashes[str(path.relative_to(root))] = reader.digest.hexdigest()


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _calls(request):
    calls = request.get("tool_calls") or []
    # An absent/non-numeric call index is ordered by its recorded position.
    return sorted(enumerate(calls), key=lambda item: (
        item[1].get("call_index") if _number(item[1].get("call_index")) else item[0], item[0]))


def _turns(session):
    turns = {}
    for req in session.get("requests") or []:
        turn = req.get("turn_index")
        if turn is None:
            raise ValueError(f"missing root turn_index in {session.get('session_id')}")
        turns.setdefault(turn, []).append(req)
    return turns.items()


def _walk(requests, path=()):
    for request in requests:
        yield path, request
        for ordinal, call in _calls(request):
            child = call.get("subagent")
            if child is not None:
                if not isinstance(child, dict) or not isinstance(child.get("requests"), list):
                    raise ValueError("invalid nested subagent")
                child_path = path + ((request.get("step"), call.get("call_index"), ordinal),)
                yield from _walk(child["requests"], child_path)


def _reason(requests):
    reasons = set()
    seen = set()
    for path, req in _walk(requests):
        identity = (path, req.get("step"))
        if req.get("step") is None or identity in seen:
            reasons.add("missing_or_duplicate_step")
        seen.add(identity)
        blocks = req.get("block_ids")
        if not isinstance(blocks, list) or not blocks:
            reasons.add("empty_blocks")
        elif len(blocks) * 2 + 128 > 32768:
            reasons.add("context_window")
    return reasons


def _stratum(requests):
    multi = mutation = child = False
    for path, req in _walk(requests):
        calls = req.get("tool_calls") or []
        child |= bool(path) or any(c.get("subagent") is not None for c in calls)
        multi |= len(calls) > 1
        mutation |= (req.get("mutation") or {}).get("transition") in ("mutation", "switch")
    return "subsession" if child else "mutation_switch" if mutation else "multi_call" if multi else "ordinary"


def _hist(counter):
    return {str(k): v for k, v in sorted(counter.items(), key=lambda item: str(item[0]))}


def profile(dataset, run_dir):
    """Stream all releases and write trace-profile.json; return the profile."""
    root, weeks, files = _sources(dataset)
    counts, lengths, lcps, depths = Counter(), Counter(), Counter(), Counter()
    transitions, causes, exclusions, waits = Counter(), Counter(), Counter(), Counter()
    week_counts = defaultdict(Counter)
    sources = {}
    seen_turns = set()
    for week, session, _source in _sessions(root, files, sources):
        counts["root_sessions"] += 1
        week_counts[week]["root_sessions"] += 1
        if not session.get("session_id") or not session.get("user_id"):
            raise ValueError("root session lacks session_id or user_id")
        session_start, session_end = session.get("start_ms"), session.get("end_ms")
        if not _number(session_start) or not _number(session_end) or session_end < session_start:
            counts["session_time_anomalies"] += 1
        last_start = {}
        previous = {}
        for path, req in _walk(session.get("requests") or []):
            counts["requests"] += 1
            week_counts[week]["requests"] += 1
            depths[len(path)] += 1
            blocks = req.get("block_ids") or []
            lengths[len(blocks) * 16] += 1
            if not blocks:
                counts["empty_blocks"] += 1
            if path in previous:
                lcp = 0
                for a, b in zip(previous[path], blocks):
                    if a != b:
                        break
                    lcp += 1
                lcps[lcp] += 1
            previous[path] = blocks
            m = req.get("mutation") or {}
            transitions[m.get("transition") or "none"] += 1
            causes[m.get("cause") or "none"] += 1
            start, end = req.get("start_ms"), req.get("end_ms")
            if not _number(start) or not _number(end) or end < start:
                counts["request_time_anomalies"] += 1
            if _number(start):
                if path in last_start and start < last_start[path]:
                    counts["nonmonotonic_request_starts"] += 1
                last_start[path] = start
            completion = (req.get("provider_tokens") or {}).get("completion")
            if not _number(completion):
                counts["missing_completion"] += 1
            if req.get("step_trigger") == "tool" and not req.get("link_reason"):
                counts["missing_tool_links"] += 1
            calls = req.get("tool_calls") or []
            counts["tool_calls"] += len(calls)
            week_counts[week]["tool_calls"] += len(calls)
            counts["multi_call_requests"] += len(calls) > 1
            for call in calls:
                if call.get("subagent") is not None:
                    counts["subsessions"] += 1
                latency = call.get("latency_ms")
                if not _number(latency) or latency < 0:
                    counts["invalid_or_missing_tool_latency"] += 1
                    if _number(latency) and latency < 0:
                        counts["negative_tool_latency"] += 1
                else:
                    waits[latency] += 1
                    counts["tool_latency_over_2000_ms"] += latency > 2000
        for turn, requests in _turns(session):
            key = (week, session["session_id"], turn)
            if key in seen_turns:
                counts["duplicate_root_turns"] += 1
                continue
            seen_turns.add(key)
            counts["root_turns"] += 1
            week_counts[week]["root_turns"] += 1
            reasons = _reason(requests)
            if reasons:
                counts["excluded_root_turns"] += 1
                exclusions.update(reasons)
            else:
                counts["eligible_" + _stratum(requests)] += 1
    result = {
        "schema_version": 1, "dataset": str(root.parent), "weeks": weeks,
        "calibration_weeks": weeks[:4], "heldout_weeks": weeks[-4:],
        "sources_sha256": sources, "counts": dict(counts), "week_counts": {week: dict(week_counts[week]) for week in weeks},
        "exclusions": {name: exclusions[name] for name in ("empty_blocks", "context_window", "missing_or_duplicate_step")},
        "context_tokens_histogram": _hist(lengths), "lcp_blocks_histogram": _hist(lcps),
        "request_depth_histogram": _hist(depths), "transitions": dict(transitions),
        "mutation_causes": dict(causes), "tool_latency_ms_histogram": _hist(waits),
        "block_tokens": 16, "discarded_partial_block": True,
    }
    _write(Path(run_dir).resolve() / "trace-profile.json", _json(result) + "\n")
    return result


def _ordinary_ids(tokenizer_path):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
    special = set(tokenizer.all_special_ids)
    special.update(i for i, token in tokenizer.added_tokens_decoder.items() if token.special)
    ids = []
    for token_id in sorted(set(tokenizer.get_vocab().values())):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if any(unicodedata.category(c) in ("Cc", "Cf") for c in text):
            continue
        ids.append(token_id)
        if len(ids) == 16384:
            break
    if len(ids) != 16384:
        raise ValueError("tokenizer has fewer than 16384 ordinary non-control IDs")
    return ids


def _chain(week, blocks):
    parent = hashlib.sha256((VERSION + "\0" + week).encode()).digest()
    for block in blocks:
        encoded = _json(block).encode()
        yield parent, encoded
        parent = hashlib.sha256(parent + len(encoded).to_bytes(8, "big") + encoded).digest()


def _pair(parent, block, ids, salt=0):
    digest = hashlib.sha256(parent + len(block).to_bytes(8, "big") + block + salt.to_bytes(8, "big")).digest()
    return (ids[int.from_bytes(digest[:8], "big") % len(ids)],
            ids[int.from_bytes(digest[8:16], "big") % len(ids)])


def _projection(candidates, ids):
    # Most trie nodes have one child; retain overrides only for collisions.
    siblings = {}
    for candidate in candidates:
        for _, request in _walk(candidate["requests"]):
            for parent, block in _chain(candidate["week"], request["block_ids"]):
                old = siblings.get(parent)
                if old is None:
                    siblings[parent] = block
                elif isinstance(old, set):
                    old.add(block)
                elif old != block:
                    siblings[parent] = {old, block}
    overrides, collisions = {}, 0
    for parent, blocks in siblings.items():
        if not isinstance(blocks, set):
            continue
        used = set()
        for block in sorted(blocks):
            salt = 0
            pair = _pair(parent, block, ids)
            while pair in used:
                salt += 1
                collisions += 1
                pair = _pair(parent, block, ids, salt)
            if salt:
                overrides[parent, block] = pair
            used.add(pair)
    return overrides, collisions


def _request_rows(week, session_id, requests, mapping, ids, counters, path=(), parent=None):
    rows = []
    previous = parent
    for req in requests:
        identity = [week, session_id, [list(p) for p in path], req["step"]]
        tokens = []
        for key in _chain(week, req["block_ids"]):
            tokens.extend(mapping[key] if key in mapping else _pair(*key, ids))
        completion = (req.get("provider_tokens") or {}).get("completion")
        if not _number(completion):
            counters["missing_completion"] += 1
            completion = 1
        max_new = max(1, min(int(completion), 128))
        tools = []
        valid_latencies = []
        for ordinal, call in _calls(req):
            latency = call.get("latency_ms")
            if not _number(latency) or latency < 0:
                counters["missing_or_invalid_tool_latency"] += 1
                latency = 0
            valid_latencies.append(latency)
            sub = call.get("subagent")
            child_path = path + ((req["step"], call.get("call_index"), ordinal),)
            children = _request_rows(week, session_id, sub["requests"], mapping, ids, counters, child_path, identity) if sub is not None else []
            tools.append({"call_index": call.get("call_index"), "ordinal": ordinal, "latency_ms": latency,
                          "subagent_session_id": sub.get("session_id") if sub is not None else None,
                          "subagent_user_id": sub.get("user_id") if sub is not None else None,
                          "subagent": children})
        raw_barrier = max(valid_latencies, default=0)
        counters["barriers_capped"] += raw_barrier > 2000
        rows.append({"identity": identity, "predecessor": previous, "input_ids": tokens,
                     "max_new_tokens": max_new, "ignore_eos": True,
                     "mutation": req.get("mutation"), "step_trigger": req.get("step_trigger"),
                     "link_reason": req.get("link_reason"), "start_ms": req.get("start_ms"),
                     "barrier_ms": min(2000, raw_barrier), "tools": tools})
        previous = identity
    return rows


def select(dataset, run_dir, tokenizer_path, split="heldout", smoke=True):
    """Select complete turns, then project their chains using the locked tokenizer."""
    if split not in ("calibration", "heldout"):
        raise ValueError("split must be calibration or heldout")
    root, weeks, files = _sources(dataset)
    run = Path(run_dir).resolve()
    profile_path = run / "trace-profile.json"
    with profile_path.open(encoding="utf-8") as f:
        prof = json.load(f)
    if prof["weeks"] != weeks or prof["dataset"] != str(root.parent):
        raise ValueError("profile does not match the dataset")
    split_weeks = set(weeks[:4] if split == "calibration" else weeks[-4:])
    limit = 16 if smoke else 64
    buckets = {s: [] for s in STRATA}
    eligible, excluded, seen = Counter(), Counter(), set()
    sources = {}
    for week, session, source in _sessions(root, ((w, p) for w, p in files if w in split_weeks), sources):
        if not session.get("session_id") or not session.get("user_id"):
            raise ValueError("root session lacks session_id or user_id")
        for turn_order, (turn, requests) in enumerate(_turns(session)):
            key = (week, session["session_id"], turn)
            if key in seen:
                excluded["duplicate_root_turn"] += 1
                continue
            seen.add(key)
            reasons = _reason(requests)
            if reasons:
                excluded.update(reasons)
                excluded["total_turns"] += 1
                continue
            stratum = _stratum(requests)
            eligible[stratum] += 1
            digest = _sha((VERSION + week + str(session["session_id"]) + str(turn)).encode())
            candidate = {"week": week, "session_id": session["session_id"], "user_id": session["user_id"],
                         "turn_index": turn, "turn_order": turn_order, "source": source, "stratum": stratum,
                         "requests": requests, "selection_hash": digest}
            bucket = buckets[stratum]
            bucket.append((digest, _json(key), candidate))
            bucket.sort(key=lambda x: (x[0], x[1]))
            if len(bucket) > limit:
                bucket.pop()
    # Non-participating weeks still have to match the whole-dataset profile,
    # but calculating their compressed source hashes does not require JSON.
    for week, path in files:
        if week not in split_weeks:
            with path.open("rb") as f:
                sources[str(path.relative_to(root))] = hashlib.file_digest(f, "sha256").hexdigest()
    if sources != prof["sources_sha256"]:
        raise ValueError("trace source hashes differ from profile")
    candidates = [entry[2] for stratum in STRATA for entry in buckets[stratum]]
    ids = _ordinary_ids(tokenizer_path)
    mapping, collisions = _projection(candidates, ids)
    candidates.sort(key=lambda c: (c["week"], c["session_id"], c["turn_order"]))
    counters, previous, sample_hashes = Counter(), {}, []
    name = f"trace-{split}-{'smoke' if smoke else 'main'}"
    run.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".trace-", dir=run)
    row_hash = hashlib.sha256()
    try:
        with os.fdopen(fd, "wb") as output:
            for candidate in candidates:
                session_key = (candidate["week"], candidate["session_id"])
                turn_identity = [candidate["week"], candidate["session_id"], candidate["turn_index"]]
                row = {k: candidate[k] for k in ("week", "session_id", "user_id", "turn_index", "turn_order", "source", "stratum", "selection_hash")}
                row["start_ms"] = candidate["requests"][0].get("start_ms")
                row["previous_selected_turn"] = previous.get(session_key)
                row["requests"] = _request_rows(candidate["week"], candidate["session_id"], candidate["requests"], mapping, ids, counters)
                encoded = (_json(row) + "\n").encode()
                output.write(encoded)
                row_hash.update(encoded)
                sample_hashes.append(row["selection_hash"])
                previous[session_key] = turn_identity
        os.replace(temporary, run / (name + ".jsonl"))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    token_files = {}
    token_path = Path(tokenizer_path).resolve()
    for path in sorted(token_path.iterdir()):
        if path.is_file() and (path.name.startswith("tokenizer") or path.name in ("vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json")):
            with path.open("rb") as f:
                token_files[path.name] = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = {"schema_version": 1, "algorithm": VERSION, "split": split, "smoke": smoke,
                "weeks": sorted(split_weeks), "profile_sha256": _sha(profile_path.read_bytes()),
                "source_sha256": sources, "tokenizer_path": str(token_path), "tokenizer_files_sha256": token_files,
                "ordinary_ids_sha256": _sha(_json(ids).encode()), "selected_rows_sha256": row_hash.hexdigest(),
                "selected_count": len(candidates), "strata": {s: {"eligible": eligible[s], "selected": len(buckets[s]),
                    "shortfall": max(0, limit - len(buckets[s])), "selection_rate": len(buckets[s]) / eligible[s] if eligible[s] else 0} for s in STRATA},
                "exclusions": dict(excluded), "counters": dict(counters), "projection_collisions": collisions,
                "sample_hashes": sample_hashes, "block_tokens": 16,
                "projected_tokens_per_block": 2, "discarded_partial_block": True,
                "barrier_rule": "min(2000,max(valid nonnegative tool latency)); max with serial child duration"}
    _write(run / (name + ".manifest.json"), _json(manifest) + "\n")
    return manifest


def main():
    import argparse
    from .gpu_run import MODEL

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("profile", "select"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=MODEL)
    parser.add_argument("--split", choices=("calibration", "heldout"), default="heldout")
    parser.add_argument("--main-sample", action="store_true")
    cfg = parser.parse_args()
    if cfg.operation == "profile":
        report = profile(cfg.dataset, cfg.run_dir)
        print(_json(report["counts"]))
    else:
        report = select(cfg.dataset, cfg.run_dir, cfg.tokenizer, cfg.split, not cfg.main_sample)
        print(_json({k: report[k] for k in ("strata", "selected_rows_sha256", "selected_count")}))


if __name__ == "__main__":
    main()
