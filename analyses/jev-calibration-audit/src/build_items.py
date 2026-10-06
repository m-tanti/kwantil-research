"""Build the labelled item files. Fixed seeds, no hand-written option descriptions.

Option descriptions are the label name with underscores replaced by spaces, and nothing
else: writing better descriptions would be tuning the vendor's input, and the audit asks
what a buyer gets from the labels they already have.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "artifacts" / "items"
SEED = 20260923


def _w(name: str, rows: list[dict]) -> None:
    ITEMS.mkdir(parents=True, exist_ok=True)
    with open(ITEMS / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(name, len(rows))


def desc(label: str) -> str:
    return label.replace("_", " ").replace("?", "").strip()


def banking77() -> None:
    ds = load_dataset("legacy-datasets/banking77")
    names = ds["test"].features["label"].names
    rows = [{"id": f"b77-{i}", "state": r["text"], "truth": names[r["label"]]}
            for i, r in enumerate(ds["test"])]
    _w("banking77", rows)
    json.dump({"labels": names, "criteria": {n: desc(n) for n in names}},
              open(ITEMS / "banking77.labels.json", "w"), indent=1)


def clinc150() -> None:
    ds = load_dataset("clinc/clinc_oos", "plus")
    names = ds["test"].features["intent"].names
    keep = [n for n in names if n != "oos"]
    rows = [{"id": f"clinc-{i}", "state": r["text"], "truth": names[r["intent"]]}
            for i, r in enumerate(ds["test"]) if names[r["intent"]] != "oos"]
    _w("clinc150", rows)
    json.dump({"labels": keep, "criteria": {n: desc(n) for n in keep}},
              open(ITEMS / "clinc150.labels.json", "w"), indent=1)


def amazon(n: int = 1000, pool: int = 20000) -> None:
    ds = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_Musical_Instruments",
                      split="full", streaming=True, trust_remote_code=True)
    cand = []
    for r in ds:
        t = (r.get("text") or "").strip()
        if 20 <= len(t) <= 1500 and r.get("rating") in (1.0, 2.0, 3.0, 4.0, 5.0):
            cand.append({"title": (r.get("title") or "").strip(), "text": t, "rating": int(r["rating"])})
        if len(cand) >= pool:
            break
    rng = random.Random(SEED)
    # stratified: 200 per star, so the ordinal scoring is not dominated by 5-star reviews
    rows = []
    for star in (1, 2, 3, 4, 5):
        s = [c for c in cand if c["rating"] == star]
        rng.shuffle(s)
        rows += s[: n // 5]
    rng.shuffle(rows)
    out = [{"id": f"amz-{i}", "state": f"Title: {r['title']}\n\nReview: {r['text']}", "truth": r["rating"]}
           for i, r in enumerate(rows)]
    _w("amazon", out)


def boolq(n: int = 1000) -> None:
    ds = load_dataset("google/boolq")["validation"]
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)
    rows = []
    for i in idx[:n]:
        r = ds[i]
        q = r["question"].strip()
        q = q[0].upper() + q[1:] + ("" if q.endswith("?") else "?")
        rows.append({"id": f"boolq-{i}", "state": r["passage"], "question": q, "truth": bool(r["answer"])})
    _w("boolq", rows)


if __name__ == "__main__":
    banking77()
    clinc150()
    amazon()
    boolq()
