"""E4b: a controlled synthetic set where the whole true distribution is known by construction.

Three blocks, all seeded:
  noul_factorial : stated p x number format x distractor length, Noul question. 9 x 3 x 3 x 2 polarities.
  choice_known   : a bag of K coloured balls with stated counts, 'which colour is drawn?'. The stated
                   composition is the true probability vector, so canonical calibration is checkable.
                   K in {2, 3, 5, 10}, 10 compositions each, with and without a distractor paragraph.
  score_known    : balls marked 1 to 5 in stated counts, 'which number is drawn?' as a Score question,
                   so the ordinal distribution is known. 30 compositions.
Added 2026-09-23 after the first results, as a descriptive extension of E4. No registered claim.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jev_client import run_many  # noqa: E402
from questions import choice_q, noul_q  # noqa: E402

ITEMS = Path(__file__).resolve().parents[1] / "artifacts" / "items"
SEED = 20260923
WORDS = ("one two three four five six seven eight nine ten").split()
COLOURS = ["red", "blue", "green", "yellow", "black", "white", "orange", "purple", "brown", "grey"]
DISTRACTOR_1 = " The bag was bought at a market in Valletta last spring."
DISTRACTOR_LONG = (" The bag was bought at a market in Valletta last spring, on a morning when the ferries "
                   "were running late and the stalls had only just opened. Its owner keeps it on a shelf "
                   "beside a stack of unread magazines, a broken clock that still shows a quarter past "
                   "four, and a postcard from a cousin who moved to Adelaide and writes twice a year. The "
                   "room faces north, so the light is even and the paint has not faded. On Sundays the "
                   "radio is on, usually a programme about gardening, and the cat sleeps on the chair "
                   "nearest the window. None of this has any bearing on the balls in the bag, which were "
                   "counted carefully by two people and written on a label tied to the drawstring.")


def fmt(k: int, n: int, form: str) -> str:
    if form == "digits":
        return f"A bag holds {k} red balls and {n - k} blue balls."
    if form == "words":
        return f"A bag holds {WORDS[k - 1]} red balls and {WORDS[n - k - 1]} blue balls."
    return f"A bag holds {n} balls; {round(100 * k / n)}% of them are red and the rest are blue."


def build():
    rows = []
    for k in range(1, 10):
        for form in ("digits", "words", "percent"):
            for dist, dtext in (("none", ""), ("short", DISTRACTOR_1), ("long", DISTRACTOR_LONG)):
                for pol in ("yes", "no"):
                    p = k / 10 if pol == "yes" else 1 - k / 10
                    rows.append({"block": "noul_factorial", "p": p, "form": form, "dist": dist, "pol": pol,
                                 "state": fmt(k, 10, form) + dtext + " One ball is drawn at random.",
                                 "question": "Is the drawn ball red?" if pol == "yes" else "Is the drawn ball blue?"})
    rng = random.Random(SEED)
    for K in (2, 3, 5, 10):
        for j in range(10):
            counts = [rng.randint(1, 9) for _ in range(K)]
            n = sum(counts)
            cols = COLOURS[:K]
            desc = ", ".join(f"{c} {col}" for c, col in zip(counts, cols))
            for dist, dtext in (("none", ""), ("long", DISTRACTOR_LONG)):
                rows.append({"block": "choice_known", "K": K, "truth": [c / n for c in counts], "labels": cols, "dist": dist,
                             "state": f"A bag holds {n} balls: {desc}.{dtext} One ball is drawn at random.",
                             "criteria": {col: f"the drawn ball is {col}" for col in cols}})
    for j in range(30):
        counts = [rng.randint(0, 6) for _ in range(5)]
        if sum(counts) == 0:
            counts[2] = 1
        n = sum(counts)
        desc = ", ".join(f"{c} marked {i + 1}" for i, c in enumerate(counts))
        rows.append({"block": "score_known", "truth": [c / n for c in counts],
                     "state": f"A bag holds {n} balls: {desc}. One ball is drawn at random.",
                     "levels": ["1", "2", "3", "4", "5"]})
    for i, r in enumerate(rows):
        r["id"] = f"syn-{i}"
    with open(ITEMS / "synthetic.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return rows


if __name__ == "__main__":
    rows = build()
    jobs = []
    for r in rows:
        m = {"set": "synthetic", "id": r["id"], "block": r["block"]}
        if r["block"] == "noul_factorial":
            jobs.append((r["state"], noul_q(r["question"]), m))
        elif r["block"] == "choice_known":
            jobs.append((r["state"], choice_q("Which colour is the drawn ball?", r["criteria"]), m))
        else:
            jobs.append((r["state"], {"q": {"type": "score", "instructions": "Which number is marked on the drawn ball?",
                                            "criteria": r["levels"]}}, m))
    print(len(jobs), "synthetic jobs")
    run_many(jobs, log="e4b_synthetic", workers=6)
