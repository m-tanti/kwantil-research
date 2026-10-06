"""Sealed forecasting cohort: pull open binary questions, ask Jev once each, seal the file.

Sources: Manifold (public API) and Polymarket (public gamma API). Metaculus now requires an
account token and is dropped (PLAN.md section 9). Filters pinned at first run:
  - created/started on or after 2026-09-15, closing on or before 2026-12-31
  - closes at least 7 days after collection (drops 5-minute and same-day markets)
  - Manifold: at least 5 unique bettors; Polymarket: volume at least 1,000 USD, Yes/No outcomes
  - excluded: question text naming an AI model, benchmark or product
Usage: python src/sealed_collect.py <cohort_number>
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent))
from jev_client import load_log, run_many  # noqa: E402
from questions import noul_q  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEALED = ROOT / "artifacts" / "sealed"
CUT_OPEN = dt.datetime(2026, 9, 15, tzinfo=dt.timezone.utc)
CUT_CLOSE = dt.datetime(2026, 12, 31, 23, 59, tzinfo=dt.timezone.utc)
AI_RE = re.compile(r"\b(gpt|claude|gemini|llama|openai|anthropic|jev|typesafe|deepseek|grok|mistral|"
                   r"benchmark|lmarena|chatbot arena|ai model|llm)\b", re.I)


def manifold(now: dt.datetime) -> list[dict]:
    out, seen = [], set()
    for sort in ("newest", "most-popular", "liquidity"):
        for offset in range(0, 2000, 1000):
            r = requests.get("https://api.manifold.markets/v0/search-markets",
                             params={"filter": "open", "contractType": "BINARY", "sort": sort,
                                     "limit": 1000, "offset": offset}, timeout=90)
            if r.status_code != 200:
                break
            page = r.json()
            if not page:
                break
            for m in page:
                if m["id"] in seen:
                    continue
                try:
                    created = dt.datetime.fromtimestamp(m.get("createdTime", 0) / 1000, dt.timezone.utc)
                    close = dt.datetime.fromtimestamp(m.get("closeTime", 0) / 1000, dt.timezone.utc)
                except (OSError, OverflowError, ValueError):
                    continue
                if not (CUT_OPEN <= created and close <= CUT_CLOSE and close >= now + dt.timedelta(days=7)):
                    continue
                if m.get("uniqueBettorCount", 0) < 5 or m.get("isResolved"):
                    continue
                if AI_RE.search(m.get("question", "")):
                    continue
                seen.add(m["id"])
                out.append({"source": "manifold", "src_id": m["id"], "url": m.get("url"),
                            "question": m["question"], "description": (m.get("textDescription") or "").strip()[:2000],
                            "created": created.isoformat(), "closes": close.isoformat(),
                            "community_p": m.get("probability"), "n_bettors": m.get("uniqueBettorCount")})
    return out


def _json_field(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except ValueError:
            return None
    return v


def polymarket(now: dt.datetime) -> list[dict]:
    out = []
    for offset in range(0, 5000, 100):
        r = requests.get("https://gamma-api.polymarket.com/markets",
                         params={"closed": "false", "active": "true", "limit": 100, "offset": offset,
                                 "start_date_min": CUT_OPEN.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "end_date_max": CUT_CLOSE.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                 "volume_num_min": 1000, "order": "volumeNum", "ascending": "false"},
                         timeout=90)
        if r.status_code != 200:
            break
        page = r.json()
        if not page:
            break
        for m in page:
            try:
                start = dt.datetime.fromisoformat(m["startDate"].replace("Z", "+00:00"))
                end = dt.datetime.fromisoformat(m["endDate"].replace("Z", "+00:00"))
            except Exception:
                continue
            if not (CUT_OPEN <= start and end <= CUT_CLOSE and end >= now + dt.timedelta(days=7)):
                continue
            if (m.get("volumeNum") or 0) < 1000:
                continue
            if _json_field(m.get("outcomes")) != ["Yes", "No"]:
                continue
            if AI_RE.search(m.get("question", "")):
                continue
            prices = _json_field(m.get("outcomePrices"))
            out.append({"source": "polymarket", "src_id": str(m["id"]),
                        "url": f"https://polymarket.com/market/{m.get('slug')}",
                        "question": m["question"], "description": (m.get("description") or "").strip()[:2000],
                        "created": start.isoformat(), "closes": end.isoformat(),
                        "community_p": float(prices[0]) if prices else None, "volume": m.get("volumeNum")})
    return out


def _state(x: dict) -> str:
    return (f"Forecasting question: {x['question']}\n\n"
            f"Resolution criteria: {x['description'] or '(none given)'}\n\n"
            f"Question closes: {x['closes'][:10]}.")


def main(cohort: int) -> None:
    now = dt.datetime.now(dt.timezone.utc)
    SEALED.mkdir(parents=True, exist_ok=True)
    prev = set()
    for p in SEALED.glob("cohort-*.items.jsonl"):
        if p.name == f"cohort-{cohort}.items.jsonl":
            continue
        for line in open(p, encoding="utf-8"):
            prev.add(json.loads(line)["src_id"])
    items = [x for x in manifold(now) + polymarket(now) if x["src_id"] not in prev]
    for i, x in enumerate(items):
        x["id"] = f"sealed{cohort}-{i}"
        x["collected"] = now.isoformat()
    with open(SEALED / f"cohort-{cohort}.items.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for x in items:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("items", len(items), {s: sum(1 for x in items if x["source"] == s) for s in ("manifold", "polymarket")})
    log = f"sealed_cohort{cohort}"
    jobs = [(_state(x), noul_q("Will this question resolve YES?"), {"set": "sealed", "id": x["id"]}) for x in items]
    run_many(jobs, log=log, workers=8)
    recs = {r["meta"]["id"]: r for r in load_log(log)}
    sealed = []
    for x in items:
        r = recs.get(x["id"])
        if not r:
            continue
        sealed.append({"id": x["id"], "source": x["source"], "src_id": x["src_id"], "url": x["url"],
                       "question": x["question"], "closes": x["closes"], "community_p": x["community_p"],
                       "model": r["response"]["model"], "ts": r["ts"],
                       "p_yes": r["response"]["answers"]["q"]["noul"], "req_hash": r["req_hash"]})
    out = SEALED / f"cohort-{cohort}.jsonl"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        for s in sealed:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    (SEALED / f"cohort-{cohort}.sha256").write_text(f"{digest}  cohort-{cohort}.jsonl\n")
    print("sealed", len(sealed), "sha256", digest)


if __name__ == "__main__":
    main(int(sys.argv[1]))
