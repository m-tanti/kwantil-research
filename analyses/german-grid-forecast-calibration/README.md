# German grid forecast calibration

Audits every quarter-hour of German (DE-LU) day-ahead system forecasts against
what actually happened: load, solar, onshore and offshore wind, over 13 months
from July 2025 to July 2026, with the Netherlands as a comparison zone.

Article: [German Grid Forecasts: Where the Uncertainty Bands Fail](https://kwantil.com/papers/german-grid-forecast-calibration/)

## What it found

Systematic midday bias, uncertainty bands that fail out of sample **including
conformal ones**, and roughly **€200,000 per 100 MW** of avoidable imbalance
exposure, concentrated in exactly the hours where being wrong is expensive.

The conformal result is the one worth reading closely. Adaptive conformal
methods are usually presented as distribution-free insurance against
miscalibration. What they actually deliver here is narrower than that and
narrower than an earlier version of this file claimed: they restore marginal
coverage, and they reduce but do not remove the tendency to miss in the
intervals that cost the most. The specimen audit puts numbers on it, with the
adaptive band's stated level inside its bootstrap interval at every level and
its miss-rate concentration in the most expensive intervals still above one.
The guarantee is an average over all hours; the cost is not.

## Data

ENTSO-E Transparency Platform. Free registration, then request API access from
your account settings to receive a token.

Raw pulls are not redistributed here. `src/pull_data.py` and
`src/pull_data_nl.py` rebuild them, and `src/assemble.py` builds the panel.

No results are shipped for this analysis: every artifact, derived outputs
included, is rebuilt by the scripts above. See `DATA.md`.

## Running

See [REPRODUCING.md](../../REPRODUCING.md#1-german-grid-forecast-calibration).

Note that running today pulls a later window than the published audit. Pass
explicit dates to reproduce the published figures.
