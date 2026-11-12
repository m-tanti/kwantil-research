# Facts for quoting: German grid forecast calibration and the specimen Vendor Band Audit

Each row is one number, with its exact units, what was measured, its sample and
uncertainty, and one approved sentence. Quote the sentence or the row, not a
paraphrase. If a number here disagrees with the article, the specimen or the
artifacts, that is a bug: `artifacts/specimen_de/*.json` wins.

Data: ENTSO-E Transparency Platform, bidding zone DE-LU, day-ahead system load
forecast and actual, 2025-07-01 to 2026-07-31, a 38,016 quarter-hour panel.
The bands need 60 days of history, so 32,252 intervals over 336 days are
scored. Prices: reBAP and day-ahead, as published. All rows are evidence, on
stated assumptions.

| # | Number | Units and what was measured | Sample and uncertainty | Approved sentence |
|---|---|---|---|---|
| 1 | 84.8% against 90% stated | Share of scored quarter-hours in which the outcome fell inside a trailing Gaussian 90% band | 32,252 intervals over 336 days; 95% day-block bootstrap interval 82.3% to 87.1% | "A Gaussian band stated at 90% on German day-ahead load covered 84.8% of quarter-hours." |
| 2 | 83.5% against 90% stated | The same for a rolling split-conformal band | Interval 81.0% to 86.0% | "The static conformal band covered 83.5% against a stated 90%, no better than the Gaussian one." |
| 3 | 89.6% against 90% stated | The same for the adaptive conformal band | Interval 87.7% to 91.4%, which includes 90% | "The adaptive band covered 89.6%, with an interval that includes its stated 90%." |
| 4 | EUR 90,042 per 100 MW mean load | Unreserved adverse exposure under the Gaussian 90% band: for each quarter-hour the outcome escaped the band, the excess beyond the band edge, priced at that interval's adverse imbalance-versus-day-ahead spread | Over the 336 scored days; portfolio errors assumed proportional to system errors; favourable spreads clipped to zero; carries no uncertainty for the price series | "Priced at reBAP, the Gaussian 90% band left about EUR 90,000 of unreserved adverse exposure per 100 MW of mean load over 336 scored days." |
| 5 | EUR 44,495 per 100 MW mean load | The same under the adaptive band | Same assumptions | "Under the adaptive band the same exposure was about EUR 44,500." |
| 6 | 1.35 times | Ratio of the Gaussian band's miss rate in the most expensive fifth of intervals (adverse spread above EUR 39.97/MWh) to its miss rate elsewhere | 95% interval 1.16 to 1.55 | "The band missed 1.35 times as often in the most expensive fifth of intervals as in the rest." |
| 7 | EUR 34.8 against EUR 24.6 per MWh | Mean adverse spread on intervals the Gaussian band missed, against the mean over all scored intervals | 32,252 intervals | "The mean adverse spread was EUR 24.6/MWh overall and EUR 34.8/MWh on the intervals the band missed." |
| 8 | 0.62 and 0.61 | Correlation and least-squares slope of the published Dutch load forecast against Dutch actual load, against 0.96 and 1.02 for the German pair | Same period; the integrity check's control | "The same check run on the Dutch public pair failed: correlation 0.62, slope 0.61. The two Dutch series are not a forecast and its outcome." |

## Do not say

- "EUR 90,000 over 13 months". The panel spans 13 months; the exposure accrues over the 336 scored days after the warm-up.
- "EUR 45,500 saved". It is the difference between rows 4 and 5 under stated assumptions, not a saving anyone realised.
- That any desk's portfolio carries this exposure. The figure scales system errors to 100 MW.
- That conformal methods fail. The static construction failed; the adaptive one held.
