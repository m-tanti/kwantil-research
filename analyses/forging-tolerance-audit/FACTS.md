# Facts for quoting: forging tolerance audit

Each row is one number, with its exact units, what was measured, its sample and
uncertainty, and one approved sentence. Quote the sentence or the row, not a
paraphrase. If a number here disagrees with the published specimen or the
artifacts, that is a bug: the artifacts win.

Source: `artifacts/specimen/*.json`. Data: "Sensor data set radial forging at
AFRC testbed v2", University of Strathclyde, doi:10.5281/zenodo.3405265, CC BY
4.0. 81 parts, 18 measured dimensions, 5 production days in 2017, a research
testbed. All rows are evidence. The data set carries no costs.

| # | Number | Units and what was measured | Sample and uncertainty | Approved sentence |
|---|---|---|---|---|
| 1 | 81 of 81 parts | Parts whose inner diameter at 55 mm lies outside its stated limits (29.7 to 30.3 mm) | Mean 29.568 mm, standard deviation 0.008 mm; lower 95% Wilson bound 95.5% | "On a public forging data set, one stated tolerance was missed by all 81 parts, by a constant offset: a specification that does not describe the process." |
| 2 | 9 of 81 parts (11.1%) | Parts exceeding the 237 mm upper length limit on at least one of 4 length measurements | Worst single measurement 6 of 81, 95% Wilson interval 3.4% to 15.2%; parts within a day are not independent | "9 of 81 parts were over the stated length limit on at least one of 4 measurements." |
| 3 | 8 of 18 tolerances | Stated tolerances whose nearest limit is more than 6 standard deviations from the mean | 81 parts each | "8 of the 18 stated tolerances were so wide that the records never tested them." |
| 4 | 73% against 90% stated | Share of parts inside a 90% range for part length built from the model's training errors, each day held out whole | 81 parts, 5 days; 95% Wilson interval 62% to 81% | "A 90% range built the usual way covered 73% of parts on days the model had not seen." |
| 5 | 94%, 2.4 times as wide | The same range built from errors on held-out days, and its median width relative to row 4 (6.38 mm against 2.62 mm) | Interval 86% to 97%; coverage was 62% on one of the 5 days | "Built from held-out days the range covered 94%, at 2.4 times the width." |
| 6 | 0.86 mm against 0.99 mm | Mean absolute error predicting part length with 14 process features, against using the average of the other days | 81 parts, leave one day out | "14 process features improved the prediction of part length by 0.13 mm on an error of 0.99 mm." |

## Do not say

- That this forge makes bad parts, or any yield or scrap rate. It is a research testbed and the tolerances may not be a customer's.
- That in-process prediction does not work. 14 features on 81 parts barely moved the error; a variable not in the records might.
- Any euro figure as a finding. The page's cost control starts from illustrative rates the reader replaces.
- That holding out rows instead of days is shown to fail here. The difference (88% against 94%) is not established on 5 days. The tool-wear analysis is where that is established.
