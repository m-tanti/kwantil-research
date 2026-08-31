# Data

No data files are committed to this repository, derived or otherwise. This
document is how you rebuild all of them.

## Source

[ENTSO-E Transparency Platform](https://transparency.entsoe.eu/), the statutory
publication platform for European electricity market data.

**Getting a token.** Register a free account, then request Restful API access
from ENTSO-E using the address you registered with. Access is granted manually,
usually within a few working days. The token then appears under
*Account Settings, Web Api Security Token*.

```bash
export ENTSOE_API_TOKEN=...
```

On Windows PowerShell, set the environment variable `ENTSOE_API_TOKEN` instead.

## Published window

`2025-07-01` to `2026-07-31`, 13 months, zones `DE_LU` and `NL`.

Running the pull without explicit dates fetches a **later** window than the
published audit, so figures will move. Pass the dates above to reproduce
published numbers.

## Rebuild

```bash
pip install pandas numpy scipy pyarrow entsoe-py

python src/pull_data.py            # DE-LU load, wind onshore/offshore, solar
python src/pull_data_nl.py         # NL comparison zone
python src/assemble.py             # builds artifacts/panel.parquet
python src/calibration.py          # coverage_results, conditional_bias, width_stats
python src/monthly_coverage.py     # monthly_coverage, the regime-break result
python src/nl_monetize.py          # imbalance exposure in euros
```

`monthly_coverage.py` writes figure JSON to `artifacts/figures` by default.
Point it elsewhere with `FIGURE_DATA_DIR`.

## What gets built

| Artifact | Size | Built by |
|---|---|---|
| `artifacts/raw/`, `artifacts/raw_nl/` | ~14 MB | the two pull scripts |
| `artifacts/panel.parquet` | ~3 MB | `assemble.py` |
| `coverage_results`, `conditional_bias`, `width_stats` | ~16 KB | `calibration.py` |
| `monthly_coverage` | ~8 KB | `monthly_coverage.py` |

## Cost and time

Free. The pull is the slow step and is rate-limited at source; budget 20 to 40
minutes for 13 months across both zones. Everything downstream is seconds.

## Licence

ENTSO-E data is re-usable under the platform terms with attribution. It is not
redistributed here, so those terms are yours to meet directly rather than
inherited through this repository.
