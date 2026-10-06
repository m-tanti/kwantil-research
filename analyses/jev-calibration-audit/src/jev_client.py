"""Thin client for the TypeSafe System One endpoint, with an append-only call log.

Every call is written to artifacts/raw/<log>.jsonl with the request body, the full response,
the model version the server reports, latency and usage. Nothing is scored from memory; the
analysis reads these logs. The model id is pinned, never an alias.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.13.0"  # pinned 2026-09-23; the alias jev-latest also resolved here on that date
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw"
_lock = threading.Lock()


def _key() -> str:
    k = os.environ.get("JEV_KEY")
    if not k:
        raise RuntimeError("JEV_KEY not set")
    return k


def body_hash(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def call(state, questions: dict, *, log: str, meta: dict | None = None, model: str = MODEL,
         retries: int = 6) -> dict:
    """One request. Returns the parsed response; appends a record to artifacts/raw/<log>.jsonl."""
    body = {"model": model, "state": state, "questions": questions}
    rec = {"ts": None, "meta": meta or {}, "req_hash": body_hash(body), "request": body}
    delay = 1.0
    for attempt in range(retries):
        t0 = time.time()
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {_key()}",
                                            "Content-Type": "application/json"},
                              json=body, timeout=90)
        except requests.RequestException as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            time.sleep(delay); delay = min(delay * 2, 30); continue
        lat = time.time() - t0
        if r.status_code in (429, 529, 500, 502, 503, 504):
            rec["error"] = f"http {r.status_code}"
            time.sleep(delay); delay = min(delay * 2, 30); continue
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec["status"] = r.status_code
        rec["latency_s"] = round(lat, 3)
        try:
            rec["response"] = r.json()
        except ValueError:
            rec["response"] = {"_raw": r.text[:2000]}
        rec.pop("error", None)
        _write(log, rec)
        if r.status_code != 200:
            raise RuntimeError(f"jev {r.status_code}: {r.text[:300]}")
        return rec["response"]
    rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec["status"] = "failed"
    _write(log, rec)
    raise RuntimeError(f"jev call failed after {retries} attempts: {rec.get('error')}")


def _write(log: str, rec: dict) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(RAW / f"{log}.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def done_hashes(log: str) -> set[str]:
    """Request hashes already answered with HTTP 200 in this log, so a rerun resumes."""
    p = RAW / f"{log}.jsonl"
    if not p.exists():
        return set()
    out = set()
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if rec.get("status") == 200:
                out.add(rec["req_hash"])
    return out


def run_many(jobs: list[tuple], *, log: str, workers: int = 8, model: str = MODEL) -> int:
    """jobs: (state, questions, meta). Skips requests already in the log. Returns count run."""
    done = done_hashes(log)
    todo = [(s, q, m) for (s, q, m) in jobs
            if body_hash({"model": model, "state": s, "questions": q}) not in done]
    print(f"[{log}] {len(jobs)} jobs, {len(jobs) - len(todo)} already done, running {len(todo)}")
    n = 0
    def _one(j):
        s, q, m = j
        try:
            call(s, q, log=log, meta=m, model=model)
            return 1
        except Exception as e:  # logged inside call()
            print("  failed:", m, str(e)[:120])
            return 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for k in ex.map(_one, todo):
            n += k
    print(f"[{log}] ran {n}/{len(todo)}")
    return n


def load_log(log: str) -> list[dict]:
    p = RAW / f"{log}.jsonl"
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("status") == 200:
                out.append(rec)
    # keep the last successful record per request hash
    last = {}
    for r in out:
        last[r["req_hash"]] = r
    return list(last.values())
