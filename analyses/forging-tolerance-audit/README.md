# Forging tolerance audit

A specimen Process Calibration Audit: every stated tolerance on a forged part
checked against what a coordinate measuring machine recorded, then a predicted
range for part length scored on production days the model never saw.

Published as a deliverable at
[kwantil.com/specimen/process-calibration-audit](https://kwantil.com/specimen/process-calibration-audit/).
It complements the [tool-wear analysis](../tool-wear-calibration/), which rests
on 16 inserts from a 1990s machine.

## What it found

81 parts, 18 measured dimensions, 5 production days.

- **No part meets every stated tolerance.** The inner diameter at 55 mm is
  outside its limits on all 81 parts, by a constant offset: mean 29.568 mm,
  standard deviation 0.008 mm, against limits of 29.7 to 30.3. That is a
  specification that does not describe the process, not a process out of
  control.
- **Part length runs at its upper limit.** 9 of 81 parts exceed 237 mm on at
  least one of the 4 length measurements. On the worst measurement 6 of 81 are
  over (95% Wilson interval 3.4% to 15.2%).
- **8 of the 18 tolerances were never tested.** The nearest limit is more than
  6 standard deviations from the mean, so those records say nothing about them.
- **A range built from training errors covered 73% against a stated 90%**
  (interval 62% to 81%). Built from errors on held-out days it covered 94%
  (86% to 97%) and was 2.4 times as wide.
- **The process features barely predict length.** Mean absolute error 0.86 mm
  with 14 features against 0.99 mm for the average of the other days.

## What it does not support

- Any production yield. This is a research testbed running a designed trial.
- Any claim about which of specification and process is wrong on the 55 mm
  diameter. Only the owner of the drawing can say.
- Independence of parts. The intervals are Wilson intervals on 81 parts; parts
  forged on one day share a setup, and 5 days are too few to resample.
- Money. The data set carries no costs. The published page converts counts to
  euros at rates the reader enters, with illustrative defaults labelled as such.

## Data

"Sensor data set radial forging at AFRC testbed v2", Tachtatzis, Gourlay,
Andonovic and Panni, University of Strathclyde,
[doi:10.5281/zenodo.3405265](https://doi.org/10.5281/zenodo.3405265), CC BY 4.0.
Not redistributed here; see [DATA.md](DATA.md).

## Running

```bash
pip install -e .[dev]
python src/features.py        # ~1 min, reads 81 sensor traces
python src/audit_forging.py   # seconds
pytest -q                     # 9 tests
```
