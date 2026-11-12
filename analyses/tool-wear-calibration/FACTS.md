# Facts for quoting: tool wear calibration audit

Each row is one number, with its exact units, what was measured, its sample and
uncertainty, and one approved sentence. Quote the sentence or the row, not a
paraphrase. If a number here disagrees with the article or the figure data,
that is a bug: `artifacts/figures/*.json` wins.

Data: NASA Milling Data Set (Agogino and Goebel, 2007). 16 inserts, 167 cuts,
146 with flank wear measured by microscope, on a 1990s machining centre. All
rows are evidence. All costs are in cuts; no currency appears in this analysis.

| # | Number | Units and what was measured | Sample and uncertainty | Approved sentence |
|---|---|---|---|---|
| 1 | 0.165 mm | Mean absolute error predicting flank wear from cut count and cutting conditions, on an insert the model never saw | 16 inserts, 146 measured cuts, leave one insert out; about a quarter of a conventional 0.60 mm change point | "Fitted as a model, a fixed change interval missed measured wear by 0.165 mm on a 0.60 mm limit." |
| 2 | 68.5% against 90% stated | Share of cuts inside a 90% range built from the model's own training errors | 95% cluster-bootstrap interval over inserts 54.7% to 82.4%; mean of 25 calibration draws | "The range a shop would quote around its change interval covered 68.5% against a promised 90%." |
| 3 | 84.4% | The same range built from errors on held-out inserts | Interval 74.4% to 93.8%, which includes 90% | "Measured on held-out inserts the range covered 84.4%, with an interval that includes 90%." |
| 4 | 55.0% on worn tools | Coverage of the held-out range on cuts with wear at or above the change point | 20 observations from 10 inserts, 6 of them from one insert | "On worn tools, the only place the range is consulted, it covered 55%." |
| 5 | About 16 points and about 6 points | Split of the 21.5-point shortfall in row 2: from measuring error on training data, and from assuming a normal shape | Same runs as rows 2 and 3 | "About 16 of the 21.5 missing points came from checking error on data the model had already seen, and about 6 from the bell curve." |
| 6 | 89.8% reported, 19% too narrow | Coverage reported by a range calibrated on held-out rows instead of held-out inserts, and how much narrower it is than the insert-held-out range | Same data | "Holding out rows instead of whole inserts reported 89.8% against 90% and was 19% too narrow." |
| 7 | 42.0 cuts per 100 | Tool life thrown away by pulling when the top of the range reaches the limit, which leaves 0.4 cuts per 100 past the limit | Policies simulated insert by insert on unseen inserts, 0.60 mm change point | "Driving over-limit cuts to nearly nothing cost about 42 cuts of tool life in every 100." |
| 8 | 44 of 167 runs | Runs in which the DC spindle-current channel clips at the converter's 10 V ceiling; median wear 0.500 mm where more than half clipped against 0.230 mm elsewhere | Exact counts | "The sensor that most directly measures cutting load clipped in 44 of 167 runs, and it clipped on the worn tools." |

## Do not say

- That these numbers transfer to another alloy, machine or shop. The article says they do not.
- That sensors or monitoring do not help. 16 inserts cannot show that they do.
- Any figure in money. The article deliberately has none.
- Row 4 without its sample: 20 observations, 6 from one insert.
