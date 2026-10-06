"""E1: every labelled item through Jev, one question per call, pinned templates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jev_client import run_many  # noqa: E402
from questions import intent_q, noul_q, stars_q  # noqa: E402

ITEMS = Path(__file__).resolve().parents[1] / "artifacts" / "items"


def read(name):
    return [json.loads(line) for line in open(ITEMS / f"{name}.jsonl", encoding="utf-8")]


def jobs_for(name: str) -> list[tuple]:
    rows = read(name)
    if name in ("banking77", "clinc150"):
        crit = json.load(open(ITEMS / f"{name}.labels.json"))["criteria"]
        return [(r["state"], intent_q(crit), {"set": name, "id": r["id"]}) for r in rows]
    if name == "amazon":
        return [(r["state"], stars_q(), {"set": name, "id": r["id"]}) for r in rows]
    if name == "boolq":
        return [(r["state"], noul_q(r["question"]), {"set": name, "id": r["id"]}) for r in rows]
    raise ValueError(name)


if __name__ == "__main__":
    sets = sys.argv[1:] or ["boolq", "amazon", "banking77", "clinc150"]
    for s in sets:
        run_many(jobs_for(s), log=f"e1_{s}", workers=8)
