"""Resumable monthly-chunked pull of DE-LU series from the ENTSO-E Transparency Platform.

Writes artifacts/raw/<series>/<YYYY-MM>.parquet; skips chunks that already exist,
so it can be re-run safely after failures. Sequential + politely paced (well under
the 400 req/min limit). Failures are logged and skipped, not fatal - rerun to retry.
"""
import os
import time
import traceback
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient
import sys


# Windows consoles default to cp1252 and die on any character outside it. This
# is a reporting script; it should not fail on the last line because a result
# contained a symbol.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw"
ZONE = "DE_LU"
TZ = "Europe/Berlin"
MONTHS = pd.period_range("2025-07", "2026-07", freq="M")

TOKEN = os.environ["ENTSOE_API_TOKEN"]
client = EntsoePandasClient(api_key=TOKEN)


def redact(text: str) -> str:
    """The ENTSO-E API carries the token as a URL query parameter, so a requests
    exception can embed it in the failing URL. Nothing prints unredacted."""
    return text.replace(TOKEN, "***")

SERIES = {
    "load_forecast": lambda s, e: client.query_load_forecast(ZONE, start=s, end=e),
    "load_actual": lambda s, e: client.query_load(ZONE, start=s, end=e),
    "wind_solar_forecast": lambda s, e: client.query_wind_and_solar_forecast(ZONE, start=s, end=e),
    "generation_actual": lambda s, e: client.query_generation(ZONE, start=s, end=e, nett=True),
    "day_ahead_prices": lambda s, e: client.query_day_ahead_prices(ZONE, start=s, end=e),
    "imbalance_prices": lambda s, e: client.query_imbalance_prices(ZONE, start=s, end=e),
}


def normalize(obj) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        obj = obj.to_frame("value")
    if isinstance(obj.columns, pd.MultiIndex):
        obj.columns = [" | ".join(map(str, c)).strip() for c in obj.columns]
    obj.index.name = "ts"
    return obj


def main() -> None:
    failures = []
    for name, fn in SERIES.items():
        outdir = RAW / name
        outdir.mkdir(parents=True, exist_ok=True)
        for m in MONTHS:
            out = outdir / f"{m}.parquet"
            if out.exists():
                continue
            s = pd.Timestamp(m.start_time, tz=TZ)
            e = pd.Timestamp((m + 1).start_time, tz=TZ)
            try:
                df = normalize(fn(s, e))
                df.to_parquet(out)
                print(f"OK   {name} {m}: {len(df)} rows", flush=True)
            except Exception as ex:
                err = redact(repr(ex))
                failures.append((name, str(m), err))
                print(f"FAIL {name} {m}: {err}", flush=True)
                print(redact(traceback.format_exc()), flush=True)
            time.sleep(1.0)

    print(f"\ndone; failures: {len(failures)}")
    for f in failures:
        print("  ", *f)


if __name__ == "__main__":
    main()
