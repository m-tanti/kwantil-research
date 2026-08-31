# Tool wear calibration audit

Audits a fixed tool-change interval the way any other model would be audited,
using 16 milling inserts that were run to failure and measured under a
microscope between cuts.

Article: [Tool Change Intervals, Audited Against Measured Wear](https://kwantil.com/papers/tool-wear-calibration-audit/)

## What it found

The fixed rule carries a leave-one-insert-out error of 0.165 mm, about a quarter
of a conventional 0.60 mm change point. The interval a shop would quote around
it covers 68.5% of outcomes against a promised 90%, and separating the two
mistakes inside that number matters: roughly 16 points of the shortfall comes
from computing error on data the model was fitted on, and only about 6 from
assuming a normal shape.

**The most transferable result is about the split.** Holding out random *rows*
instead of whole *inserts* leaves cuts from the same edge on both sides. That
reports 89.8% coverage against a nominal 90%, passes any marginal check anyone
runs, and is 19% too narrow while under-covering exactly where the decision is
made. It fails by looking confident. `src/review_checks.py` demonstrates it.

## Data

NASA Milling Data Set (Agogino and Goebel, 2007; BEST Lab, UC Berkeley),
distributed by the NASA Prognostics Center of Excellence. 98 MB of MATLAB files,
not redistributed here. Place `mill.mat` under `artifacts/raw/`.

Shipped: the figure JSON under `artifacts/figures/`, so the published charts can
be checked without running the pipeline.

## Running

See [REPRODUCING.md](../../REPRODUCING.md#3-tool-wear-calibration). `calibrate.py`
is the long step: 6 constructions × 4 feature sets × 25 seeds × 16 folds.
