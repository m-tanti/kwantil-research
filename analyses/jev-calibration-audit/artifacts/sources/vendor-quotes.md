# Vendor statements, as read on 2026-09-23

Each quote was read on the date above from the page named. The plan (PLAN.md section 2)
required these before any call was scored.

## What "calibrated" means to the vendor

Launch post, https://typesafe.ai/blog/introducing-system-one-models-and-jev, dated
2026-09-15:

> Always communicates confidence and uncertainty with every output. Calibrated: higher
> confidence means higher accuracy.

> answers with epistemically honest probabilities on System One tasks

> More consistent: returns similar answers for similar inputs

The training method is named as "Reinforcement Learning for Calibrated Decisions (RLCD)".
No paper, calibration curve or training detail was published as of the read date.

## How the vendor's benchmarks are scored

Same post:

> We use the average of GPT-6 Astra and Fable 5.1 as the reference answer

The launch benchmarks therefore measure agreement with a two-model consensus, not
agreement with outcomes. The vendor's capabilities team wrote the workflows tested
(MarkTechPost, 2026-09-19, https://www.marktechpost.com/2026/09/19/typesafe-ai-releases-jev/).

## The two uncertainty outputs

Confidence page, https://docs.typesafe.ai/confidence:

> The shape of that distribution is what tells you how certain the model is: concentrated
> on one outcome means a confident answer, spread out means an uncertain one.

The page gives, for a three-option demo, the approximation "(3 x largest probability - 1) / 2".
Confidence applies to Choice and Score answers; "Noul answers don't carry one".

> Start with conservative thresholds, test with your own data, and adjust as you observe
> results.

Noul page, https://docs.typesafe.ai/primitives/noul:

> [The returned value] runs from 0 to 1, but it's not a scale of the thing you asked about.
> It is the probability that the answer is yes.

> Where to set the threshold depends on the cost of being wrong. Use 0.5 when yes and no
> are equally easy to act on. Raise it when acting on a false yes is expensive. Lower it
> when missing a true yes is expensive.

## Model versions

Models page, https://docs.typesafe.ai/models: `jev-1.13.0` is the current production
version; `jev-latest` and `jev-preview` both resolved to it on the read date. The response
`model` field returns the versioned id that served the request. The page recommends
pinning a version once thresholds have been calibrated against it. Every call in this
audit was made with `model: "jev-1.13.0"` and every response reported that id.

## Known weak spots (the vendor's own list)

Jaggedness page for jev-1.13, https://docs.typesafe.ai/model-jaggedness/jev-1.13,
"last reviewed 2026-09-17". Eleven entries: literal reading; counting; numeric
representations; score interpolation; date and time comparison; indirection and double
negatives; irrelevant context as a distractor; adversarial content; contradictory criteria;
structural invariants (no guarantee that P(noul) + P(not noul) = 1 across separate
questions); text generation.

## Terms

Master Customer Agreement, https://typesafe.ai/legal/mca, last updated 2026-09-19. No
clause restricts publishing evaluations or benchmarks. Section 2.3(b) prohibits using
outputs to train or distil an imitating model, and using the service to develop a
competing product. Neither applies to this audit; no output here trains anything.

## Pricing

Third-party price pages disagree: $0.042 per million input tokens (Requesty) against
$0.25 to $0.42 (jevtypesafeai.com). Output tokens are free on both. The vendor's own
figure is about $0.0004 per decision. The audit's roughly 11,000 calls cost between $0.20
and $4.40 on any of these.

## Prior independent work, read 2026-09-23

- priorbench/jev (GitHub): 5,721 calls, 21 experiments, 50 pre-registered predictions
  (26 confirmed, 21 falsified), 400-item reference set, model string
  `typesafe/jev-1.13-20260917` via OpenRouter. Reports accuracy flat across confidence
  from 0.50 to 0.95. A summary of it (beri.net) cites ECE 0.107 against a noise floor and
  fitted temperatures of 0.66 for Noul and 3.29 and 3.40 for Choice and Score.
- anisselbd/jev-phishing-bench (GitHub): 2,000 synthetic phishing emails; 62.6% asked
  once, 95.0% when decomposed into 5 questions and combined by logistic regression.
- ejs-5/jev-benchmark (GitHub): 868 decisions from n8n commits with mechanical labels;
  accuracy 37% to 86% by task; confidence-binned accuracy tables show Noul under-confident
  and multi-class over-confident. No ECE reported.

None of these scores Jev against outcomes that were unknown when the prediction was made.
