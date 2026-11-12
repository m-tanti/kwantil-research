# Facts for quoting: tail-risk backtest and the specimen Model Health Check

Each row is one number, with its exact units, what was measured, its sample and
uncertainty, and one approved sentence. Quote the sentence or the row, not a
paraphrase. If a number here disagrees with the article, the specimen or the
artifacts, that is a bug: the artifacts win.

Sources: `experiments/factorial/*/as_null_sensitivity.csv`,
`artifacts/health_check/*.json`. Data: SPY, TLT, GLD daily closes, 2,172 graded
trading days, 2018-01-02 to 2026-08-24. Evidence unless marked ILLUSTRATION.

| # | Number | Units and what was measured | Sample and uncertainty | Approved sentence |
|---|---|---|---|---|
| 1 | 0 of 10, then 10 of 10 | Model-and-level cells (5 models at 2 confidence levels) whose conditional expected-shortfall statistic (Acerbi-Székely Z1) passes | Gaussian reference against Student-t with 5 degrees of freedom; models, data, VaR and ES held fixed; 2,000 simulated paths per reference, seeded; 17 to 79 breach days per cell | "Re-scored against 6 reference distributions with everything else held fixed, the same 5 models went from 0 of 10 passes to 10 of 10." |
| 2 | About 11 times | Ratio of the shift in the pass threshold caused by changing the reference (0.104) to the shift in the statistic caused by changing the model (0.0096) | One model (Conformal-PID) at the 99% level; point values from seeded runs; no interval on the ratio | "Separating a model change from a test change showed the test had moved about 11 times as far as the model." |
| 3 | 6 of 10 under every parametric reference | Cells passing the unconditional statistic (Z2) | Same sweep; 4 verdicts change between yellow and red, none between pass and fail | "The unconditional statistic gave the same 6 passes out of 10 under all 5 parametric references." |
| 4 | p = 0.18 and p = 0.24 | Z2 p-values for a model built with the right breach count and a deliberately thin tail, at the 99% and 97.5% levels | 24 and 53 breach days | "A model built with a deliberately thin tail and the correct breach count passed the unconditional test at both levels." |
| 5 | 2.07% against 1% stated | Share of trading days on which the Parametric EWMA 99% VaR was breached | 45 breaches in 2,172 days; 95% month-block bootstrap interval 1.43% to 2.77% | "A parametric 99% VaR was breached on 2.07% of 2,172 days, twice its stated rate." |
| 6 | 54% of windows | Share of complete rolling 250-day windows in which that model showed 5 or more overshootings | 1,923 overlapping windows, 8 non-overlapping; interval 28% to 86%; a correctly stated model would show 10.8% | "It spent 54% of rolling 250-day windows at 5 or more overshootings, against about 11% expected of a correct model." |
| 7 | EUR 1.62 million | ILLUSTRATION. Average rise in the VaR component of the own funds requirement caused by the backtesting addend, on an assumed EUR 100 million book | Interval EUR 0.63 million to EUR 2.64 million; book value assumed; VaR component only; one-day VaR scaled to 10 days by the square root of time; rule: Regulation (EU) No 575/2013 Articles 364 to 366 in the version in force on 8 July 2024, applicable until 1 January 2027; applies only to firms with internal-model permission | "As an illustration on an assumed EUR 100 million book, the breach record added about EUR 1.6 million on average to the VaR component of the own funds requirement." |
| 8 | EUR 3.5 million less | ILLUSTRATION. How much lower that model's average VaR component stayed, addend included, than the model that held its level | Same assumptions as row 7 | "Even with the penalty, the understating model carried about EUR 3.5 million less than the model that held its level." |
| 9 | 1,880 of 2,172 days | Days on which the first release of the joint VaR and ES model stated an expected shortfall shallower than its own VaR, at the 97.5% level (463 of 2,172 at 99%) | Exact count; fixed in the model, kept as the specimen's control | "The report's own integrity check found that one of my models stated an impossible VaR and ES pair on 1,880 of 2,172 days; it is fixed, and the old run is kept as the control." |

## Do not say

- That any real firm paid or would pay rows 7 and 8. They are illustrations.
- That expected-shortfall backtests are unreliable in general. The weakness is in a moment-matched reference this implementation chose.
- "Basel traffic light" for the expected-shortfall zones. Those zones are this analysis's own.
- Row 7 without the words "illustration" or "assumed".
