# Jev calibration audit

Audits the probabilities returned by TypeSafe's Jev (`jev-1.13.0`) against outcomes, with the
claims and thresholds registered before the first call. About 11,000 calls, 4 public labelled
sets, a synthetic block with a stated chance as the truth, and a sealed set of 372 live
forecasting questions committed by hash before any resolves.

Article: [Jev's Probabilities, Audited: Honest When Asked Yes or No, Overconfident When Asked to
Choose](https://kwantil.com/papers/jev-calibration-audit/)

## What it found

Calibration depends on the question type more than on the domain. As a multiple choice over 77
intents the model states 0.89 on average and is right 79.7% of the time (top-label ECE 0.090,
95% interval 0.078 to 0.103); over 5 ordered levels it is worse (0.151); as a yes-or-no
question it is right more often than it says (ECE 0.040, underconfident in the middle).

The two forms mean different things. Asked whether a ball drawn from a stated bag is red, the
Noul answer returns the stated chance to within 0.033 on average, and neither the number format
nor a paragraph of distraction moves it. Asked which colour the ball is, the Choice answer
overstates the likeliest colour by 0.26 to 0.50. The separate confidence field is
(K x max p - 1)/(K - 1) to within rounding and adds nothing to error detection.

Reordering or renaming the options changes the chosen answer on 3 to 4% of messages, against
0.4% on identical repeats. Offering 5 options instead of 77 raises the stated probability on the
true intent from 0.80 to 0.96 and accuracy from 84% to 96%.

One temperature fitted on 100 labelled items cuts the top-label ECE from 0.094 to 0.035. The
same repair raises the realised cost of a fraud-flagging decision from 22 to 32 per 1,000 at a
1:10 cost ratio, because that decision sums probabilities over options that are not the top
label, and flattening the vector pushes mass onto them.

**The most transferable result is the one about senses.** A vendor's "calibrated" is a claim
about the top label; a threshold on any other option, or on a sum of options, is a different
sense, and a repair aimed at the first does not repair the second. `src/run_e8_e10.py`
demonstrates it.

## Pre-registration

`content/articles/JevCalibration/claims.yaml` in the site repository holds the 6 claims, with
probabilities and resolution methods pinned on 2026-09-23. Five resolved on 2026-09-24: 3 went
against the registered expectation. JC-01, the outcome calibration on the sealed set, is scored
on 2027-01-31.

## Data

See [DATA.md](DATA.md). Nothing raw is committed except the item texts, the call logs and the
sealed cohorts, which are the audit trail.

## Running

Set `JEV_KEY`, then run the scripts in the order in DATA.md. The whole programme is about
11,000 calls, which the API answers in roughly 25 minutes at 8 concurrent requests, and costs
under $5 at any of the published prices.
