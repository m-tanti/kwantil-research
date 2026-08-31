# Data

No raw data is committed to this repository. The only committed derivatives
are the figure JSONs under `artifacts/figures/`, one of which
(`fig1_insert_life.json`) carries the measured flank-wear values it plots —
public domain, see Licence below. This document is how you rebuild everything
else.

## Source

**NASA Milling Data Set** (Agogino, A. and Goebel, K., 2007), BEST Lab, UC
Berkeley, distributed through the NASA Prognostics Center of Excellence data
repository.

**Download:** https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip
(14.7 MB zip, mirrored by the PHM Society from the NASA PCoE repository).

```bash
mkdir -p artifacts/raw
curl -L -o milling.zip "https://phm-datasets.s3.amazonaws.com/NASA/3.+Milling.zip"
unzip -j milling.zip '*mill.mat' -d artifacts/raw/
```

The zip contains a `3. Milling/` folder holding `mill.mat` (69 MB, MATLAB v5)
and a `Readme.pdf` with the experimental protocol. The scripts search for
`mill.mat` anywhere under `artifacts/raw/`, so keeping the enclosing folder or
extracting the file on its own both work.

Not redistributed here.

## What is in it

A Matsuura MC-510V machining centre, 70 mm face mill, 6 coated carbide inserts,
constant 200 m/min. Every combination of 2 materials (cast iron, J45 stainless),
2 depths of cut (0.75 and 1.5 mm) and 2 feeds (0.25 and 0.5 mm/rev), each run
twice with a fresh set.

**16 inserts, 167 cuts, 146 with flank wear measured by microscope between
cuts.** 10 of the 16 reached 0.60 mm of wear; 6 stopped short.

The measured wear is what makes the audit possible. Most tool-wear data records
when somebody *decided* to change a tool, which is a judgement rather than an
outcome, and a prediction cannot be audited against a past judgement.

## Rebuild

```bash
pip install numpy scipy pandas scikit-learn

python src/features.py             # windowed features; flags clipped channels
python src/characterise.py         # per-insert wear trajectories
python src/calibrate.py            # 6 interval constructions, leave-one-insert-out
python src/decision.py             # change-policy simulation
python src/ablation.py             # sensitivity to the wear limit
python src/review_checks.py        # demonstrates the row-split failure
python src/export_figures.py       # writes artifacts/figures/*.json
```

`calibrate.py` is the long step: 6 constructions by 4 feature sets by 25 seeds
by 16 leave-one-insert-out folds.

## What gets built

| Artifact | Size | Built by |
|---|---|---|
| `artifacts/raw/mill.mat` | 69 MB | you, from the link above (14.7 MB download) |
| `artifacts/interim/` | ~5 MB | `features.py`, `characterise.py` |
| `artifacts/figures/*.json` | ~40 KB | `export_figures.py` |

## Cost and time

Free. `features.py` takes a few minutes over the raw signals. `calibrate.py` is
the longest at tens of minutes depending on cores. Everything else is seconds.

## A note on the unit

Nine to twenty successive cuts share one cutting edge, so 146 cuts are nowhere
near 146 independent observations. Every construction here holds out whole
**inserts**. Holding out rows instead leaves cuts from the same edge on both
sides of the split and produces intervals that look well calibrated and are not.
`src/review_checks.py` reproduces that failure deliberately.

## Licence

US Government work, public domain, with attribution to the NASA Prognostics
Center of Excellence requested. The dataset is not redistributed here, apart
from the measured flank-wear values carried in
`artifacts/figures/fig1_insert_life.json`, which the public-domain status
permits.
