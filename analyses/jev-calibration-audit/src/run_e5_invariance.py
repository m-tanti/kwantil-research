"""E5: 300 Banking77 items, 6 perturbations one at a time. The noise floor (repeats) runs first.

Substitution recorded in PLAN.md section 9: 'paraphrase' is replaced by a surface-form
perturbation (lower-case, trailing punctuation stripped), because producing 300 paraphrases
would require a second language model and the audit is of one model alone.
"""
from __future__ import annotations

import json
import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jev_client import run_many  # noqa: E402
from questions import intent_q  # noqa: E402

ITEMS = Path(__file__).resolve().parents[1] / "artifacts" / "items"
SEED = 20260923
DISTRACTOR = " The weather in Valletta is mild for the time of year."


def letter(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = string.ascii_uppercase[r] + s
    return s


def build():
    rows = [json.loads(line) for line in open(ITEMS / "banking77.jsonl", encoding="utf-8")]
    lab = json.load(open(ITEMS / "banking77.labels.json"))
    labels, crit = lab["labels"], lab["criteria"]
    rng = random.Random(SEED)
    idx = list(range(len(rows)))
    rng.shuffle(idx)
    sample = [rows[i] for i in idx[:300]]
    with open(ITEMS / "e5_sample.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    jobs = []
    for r in sample:
        base = intent_q(crit)
        m = {"set": "e5", "id": r["id"]}
        n = int(r["id"].split("-")[1])
        # repeats: 5 identical calls. An identical body would be de-duplicated by the log, so
        # the question key carries the repeat index; content is unchanged.
        for k in range(5):
            jobs.append((r["state"], {f"q{k}": base["q"]}, {**m, "pert": "repeat", "rep": k}))
        # option order
        order = labels[:]
        random.Random(SEED + n).shuffle(order)
        jobs.append((r["state"], intent_q({x: crit[x] for x in order}), {**m, "pert": "order"}))
        # label names: keys become letters, descriptions unchanged
        lettered = {letter(i): crit[x] for i, x in enumerate(labels)}
        jobs.append((r["state"], intent_q(lettered),
                     {**m, "pert": "letters", "map": {letter(i): x for i, x in enumerate(labels)}}))
        # surface form
        s = r["state"].lower().rstrip(" .?!")
        jobs.append((s, base, {**m, "pert": "surface"}))
        # distractor
        jobs.append((r["state"] + DISTRACTOR, base, {**m, "pert": "distractor"}))
        # cardinality 5 and 20: the true label plus seeded others
        others = [x for x in labels if x != r["truth"]]
        rr = random.Random(SEED + 7 * n)
        for card in (5, 20):
            pick = rr.sample(others, card - 1) + [r["truth"]]
            rr.shuffle(pick)
            jobs.append((r["state"], intent_q({x: crit[x] for x in pick}), {**m, "pert": f"card{card}"}))
    return jobs


if __name__ == "__main__":
    jobs = build()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which == "noise":
        jobs = [j for j in jobs if j[2]["pert"] == "repeat"]
    run_many(jobs, log="e5_invariance", workers=8)
