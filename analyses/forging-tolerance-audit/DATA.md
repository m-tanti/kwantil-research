# Data

No raw data is committed. What is committed is `artifacts/specimen/`: the
findings register, the evidence tables and the lineage block, a few tens of
kilobytes of aggregates plus one row per part of measured and predicted length.

## Source

**Sensor data set radial forging at AFRC testbed v2.** Christos Tachtatzis,
Gordon Gourlay, Ivan Andonovic and Omer Panni, University of Strathclyde,
Advanced Forming Research Centre. Published on Zenodo, 11 September 2019.

- DOI: https://doi.org/10.5281/zenodo.3405265
- Licence: Creative Commons Attribution 4.0 International
- One archive, 282,484,853 bytes, md5 `549e0e2f365c51748373f265dff7b268`

```bash
mkdir -p artifacts/raw
curl -L -o v2.zip "https://zenodo.org/api/records/3405265/files/STRATH%20radial%20forge%20dataset%20v2.zip/content"
unzip -j v2.zip -d artifacts/raw/
```

The archive holds 81 sensor traces (`Scope0001.csv` to `Scope0081.csv`, about
1 GB unpacked), `CMMData.xlsx` with the 18 measured dimensions and the nominal
and tolerance rows, and `ForgedPartDataStructureSummaryv3.xlsx`.

## What is in it

A GFM SKK10/R radial forge. 81 hollow parts, each heated by induction, forged
in about 37 seconds, then measured on a coordinate measuring machine. The
measurement dates run from 31 October to 6 November 2017 over 5 days, although
the published description says one day of operation.

## What gets built

| Artifact | Built by | Committed |
|---|---|---|
| `artifacts/raw/` | you, from the link above | no |
| `artifacts/interim/parts.csv`, `tolerances.json` | `src/features.py` | no |
| `artifacts/specimen/*` | `src/audit_forging.py` | yes |

## Licence note

CC BY 4.0 permits reuse and redistribution with attribution. The per-part
lengths in `artifacts/specimen/predictions.json` are derived from the data set
and are shipped under that licence with the attribution above.
