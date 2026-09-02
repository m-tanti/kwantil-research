"""Assemble artifacts/raw_de/ in the same shape the NL pull produces.

The specimen is meant to be `audit_specimen.py --zone DE` and nothing else, so
this writes the four series that build_panel() expects, under the same directory
layout and column names as pull_data_nl.py.

Two of the four already exist. Load forecast, load actual and the day-ahead
price were pulled for the paper and live in artifacts/panel.parquet, so they are
re-shaped here rather than re-fetched: the Transparency Platform is not needed
for them and, at the time of writing, is not answering.

The fourth is reBAP, Germany's uniform imbalance price, which is not on the
Transparency Platform in usable form and comes instead from the four TSOs:

    GET https://ds.netztransparenz.de/api/v1/data/NrvSaldo/reBAP/Qualitaetsgesichert
        ?dateFrom=yyyy-MM-ddTHH:mm:ss&dateTo=yyyy-MM-ddTHH:mm:ss

behind OAuth2 client credentials. Set NETZTRANSPARENZ_CLIENT_ID and
NETZTRANSPARENZ_CLIENT_SECRET, issued through the OAuth Manager in the
netztransparenz extranet. There is no anonymous tier.

reBAP is one price for both directions, unlike the Dutch pair. It is written to
both the Long and Short columns so the adverse-spread calculation in
audit_specimen.py needs no zone-specific branch: a balancing party is exposed to
the spread in whichever direction is against it, and with a single price that is
the same number either way.

"Qualitaetsgesichert" is the quality-assured series, published with a lag of
some weeks. The coverage assertion below is not a formality: if the tail of the
audit window is not yet published, the audit must not silently score a short
period.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd
import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw_de"
PANEL = ROOT / "artifacts" / "panel.parquet"

TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"
API = "https://ds.netztransparenz.de/api/v1/data/NrvSaldo/reBAP/Qualitaetsgesichert"

# The audit window, matching the paper and the NL leg.
START = pd.Timestamp("2025-07-01 00:00", tz="Europe/Berlin")
END = pd.Timestamp("2026-08-01 00:00", tz="Europe/Berlin")


def token() -> str:
    cid = os.environ.get("NETZTRANSPARENZ_CLIENT_ID")
    secret = os.environ.get("NETZTRANSPARENZ_CLIENT_SECRET")
    if not (cid and secret):
        raise SystemExit(
            "NETZTRANSPARENZ_CLIENT_ID and NETZTRANSPARENZ_CLIENT_SECRET are not set.\n"
            "Register a client in the OAuth Manager of the netztransparenz extranet\n"
            "(https://extranet.netztransparenz.de); there is no anonymous access.")
    r = requests.post(TOKEN_URL, timeout=60,
                      data={"grant_type": "client_credentials",
                            "client_id": cid, "client_secret": secret})
    r.raise_for_status()
    return r.json()["access_token"]


def parse_rebap(text: str) -> pd.DataFrame:
    """Parse the semicolon CSV the endpoint returns.

    German conventions throughout: semicolon separator, comma decimal, dotted
    date. Split out and unit-tested because a silent mis-parse here would move
    every euro figure in the report and nothing downstream would notice.
    """
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str).rename(columns=str.strip)
    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        raise KeyError(f"none of {names} in {list(df.columns)}")

    date_c, from_c = pick("datum", "date"), pick("von", "from", "uhrzeit")
    val_c = next((c for c in df.columns
                  if "rebap" in c.lower() or "eur" in c.lower() or "wert" in c.lower()),
                 df.columns[-1])

    ts = pd.to_datetime(df[date_c].str.strip() + " " + df[from_c].str.strip(),
                        format="mixed", dayfirst=True)
    vals = (df[val_c].str.strip()
            .str.replace(".", "", regex=False)      # thousands
            .str.replace(",", ".", regex=False)     # decimal
            .astype(float))
    out = pd.DataFrame({"reBAP": vals.to_numpy()},
                       index=pd.DatetimeIndex(ts).tz_localize(
                           "Europe/Berlin", ambiguous="infer", nonexistent="shift_forward"))
    return out[~out.index.duplicated(keep="first")].sort_index()


def fetch_rebap() -> pd.DataFrame:
    hdr = {"Authorization": f"Bearer {token()}"}
    frames = []
    for period_start in pd.date_range(START, END, freq="MS", tz="Europe/Berlin"):
        period_end = min(period_start + pd.offsets.MonthBegin(1), END)
        if period_start >= END:
            break
        r = requests.get(API, headers=hdr, timeout=180, params={
            "dateFrom": period_start.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S"),
            "dateTo": period_end.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S")})
        r.raise_for_status()
        part = parse_rebap(r.text)
        print(f"  {period_start:%Y-%m}: {len(part):,} rows")
        frames.append(part)
    return pd.concat(frames).sort_index()


def write(series: pd.DataFrame, name: str) -> None:
    d = RAW / name
    d.mkdir(parents=True, exist_ok=True)
    for month, g in series.groupby(series.index.tz_convert("UTC").to_period("M")):
        g.to_parquet(d / f"{month}.parquet")
    print(f"  wrote {name}: {len(series):,} rows, {series.index.min()} to {series.index.max()}")


def main() -> None:
    if not PANEL.exists():
        raise SystemExit(f"{PANEL} is missing; run assemble.py first")
    panel = pd.read_parquet(PANEL)
    panel.index = pd.DatetimeIndex(panel.index).tz_convert("UTC")

    print("reshaping load and day-ahead price from the existing panel")
    write(panel[["load_fc"]].rename(columns={"load_fc": "Forecasted Load"}), "load_forecast")
    write(panel[["load_act"]].rename(columns={"load_act": "Actual Load"}), "load_actual")
    write(panel[["da_price"]].rename(columns={"da_price": "Day-ahead Price"}), "day_ahead_prices")

    print("fetching reBAP from netztransparenz")
    rebap = fetch_rebap()

    # One price, both directions. See the module docstring.
    imb = pd.DataFrame({"Long": rebap["reBAP"], "Short": rebap["reBAP"]},
                       index=rebap.index).tz_convert("UTC")

    # The quality-assured series lags publication by weeks. Scoring a window the
    # prices do not cover would quietly shorten the audit instead of failing.
    need = panel.index.max()
    if imb.index.max() < need:
        raise SystemExit(
            f"reBAP covers only to {imb.index.max()}, the audit window runs to {need}.\n"
            f"The quality-assured series has not caught up. Either wait, or shorten the\n"
            f"window deliberately and say so in the report.")
    covered = imb.reindex(panel.index).notna().mean().iloc[0]
    if covered < 0.99:
        raise SystemExit(f"reBAP covers only {covered:.1%} of the panel's quarter-hours")
    print(f"  coverage against the panel: {covered:.2%}")
    write(imb, "imbalance_prices")
    print(f"\nartifacts/raw_de is ready. Next: audit_specimen.py --zone DE")


if __name__ == "__main__":
    main()
