"""E4: known-probability probes. 4 mechanisms x 9 probabilities x 3 denominators x 2 polarities,
200 drawn with the pinned seed."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from jev_client import run_many  # noqa: E402
from questions import noul_q  # noqa: E402

ITEMS = Path(__file__).resolve().parents[1] / "artifacts" / "items"
SEED = 20260923


def build() -> list[dict]:
    out = []
    for den in (10, 20, 100):
        for k10 in range(1, 10):
            k = k10 * den // 10
            p = k / den
            for polarity in ("yes", "no"):
                ps = p if polarity == "yes" else 1 - p
                yes = polarity == "yes"
                out.append({"mech": "balls", "p": ps, "den": den, "pol": polarity,
                            "state": f"A bag holds {k} red balls and {den - k} blue balls. "
                                     f"One ball is drawn at random.",
                            "question": "Is the drawn ball red?" if yes else "Is the drawn ball blue?"})
                out.append({"mech": "die", "p": ps, "den": den, "pol": polarity,
                            "state": f"A fair {den}-sided die with faces numbered 1 to {den} is rolled once.",
                            "question": f"Is the result {k} or lower?" if yes else f"Is the result higher than {k}?"})
                out.append({"mech": "cards", "p": ps, "den": den, "pol": polarity,
                            "state": f"A shuffled deck of {den} cards contains {k} aces and {den - k} kings. "
                                     f"One card is dealt from the top.",
                            "question": "Is the dealt card an ace?" if yes else "Is the dealt card a king?"})
                pct = round(100 * k / den)
                out.append({"mech": "coin", "p": ps, "den": den, "pol": polarity,
                            "state": f"A weighted coin lands heads {pct}% of the time and tails "
                                     f"{100 - pct}% of the time. It is flipped once.",
                            "question": "Does it land heads?" if yes else "Does it land tails?"})
    rng = random.Random(SEED)
    rng.shuffle(out)
    out = out[:200]
    for i, r in enumerate(out):
        r["id"] = f"probe-{i}"
    ITEMS.mkdir(parents=True, exist_ok=True)
    with open(ITEMS / "probes.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    return out


if __name__ == "__main__":
    rows = build()
    run_many([(r["state"], noul_q(r["question"]), {"set": "probes", "id": r["id"]}) for r in rows],
             log="e4_probes", workers=8)
