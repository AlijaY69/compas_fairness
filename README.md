# Fairness interventions on COMPAS recidivism predictions

A from-scratch comparison of fairness interventions for a logistic-regression
recidivism classifier trained on ProPublica's COMPAS data. All group metrics,
the reweighing weights and the equalized-odds post-processing are implemented
by hand rather than with a fairness library.

The project started from a course assignment template. I rebuilt it to understand
how to optimize for fairness rather than for performance alone. A planned
extension uses the same data to test an LLM as a classifier: whether its
predictions change when a protected attribute is masked or swapped, and how its
group-wise error rates compare with those of the classifiers here.

## Approaches

1. **Logistic regression** with race as a feature (baseline).
2. **Fairness through unawareness**: the same model with race removed.
3. **Reweighing (pre-processing)**: each (race, label) cell in the training data is
   weighted by P(race) · P(label) / P(race, label), so race and label are
   independent in the weighted data (Kamiran & Calders, 2012).
4. **Equalized-odds post-processing**: classifier 1's predictions are randomly
   flipped with group-specific probabilities p(ŷ, race) = P(ỹ = 1 | ŷ, race).
   Instead of solving the linear program from Hardt et al. (2016), 5,000 random
   parameter sets are searched; the selected set has the smallest equalized-odds
   gap (|ΔTPR| + |ΔFPR|) among sets whose F1 is within 0.1 of the baseline.

Classifiers 1 and 2 are also evaluated on the four race × sex subgroups.

## Setup and usage

```bash
pip install -r requirements.txt
python run_experiments.py   # downloads the data, prints all metrics, writes results/
python -m pytest            # sanity checks for the metrics and reweighing code
```

## Data and metrics

The data is ProPublica's `compas-scores-two-years.csv`, filtered as in their
original analysis and restricted to African-American and Caucasian defendants.
Features: sex, age category, race, juvenile felony/misdemeanour/other counts,
prior count, charge degree, and sentence length (days between jail entry and release).
Train/test split: 80/20, seed 88.

The favourable outcome is **no recidivism** (`two_year_recid = 0`), and it is
treated as the positive class throughout:

- **Positive rate (PR)**: share of a group predicted to not reoffend.
  Statistical parity holds when the positive rates are equal across groups;
  the **statistical parity difference** is PR(Caucasian) − PR(African-American).
- **TPR**: share of people who did not reoffend who were predicted not to reoffend.
  Equal opportunity holds when TPRs are equal across groups.
- **FPR**: share of people who did reoffend who were still predicted not to reoffend.
  Equalized odds requires both TPR and FPR to be equal across groups.

## Results

| Classifier | Accuracy | Precision | Recall | F1 | PR AA | PR CC | SP diff (CC − AA) | TPR AA | TPR CC | FPR AA | FPR CC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1. Logistic regression | 0.675 | 0.668 | 0.774 | 0.717 | 0.485 | 0.799 | 0.314 | 0.688 | 0.862 | 0.312 | 0.691 |
| 2. Without race | 0.677 | 0.671 | 0.768 | 0.717 | 0.492 | 0.772 | 0.280 | 0.695 | 0.844 | 0.318 | 0.648 |
| 3. Reweighing | 0.669 | 0.662 | 0.770 | 0.712 | 0.581 | 0.671 | 0.090 | 0.779 | 0.761 | 0.411 | 0.519 |
| 4. Equalized-odds post-processing | 0.563 | 0.564 | 0.783 | 0.656 | 0.720 | 0.760 | 0.040 | 0.779 | 0.786 | 0.670 | 0.716 |

Subgroup results for classifiers 1 and 2 are in [`results/intersectional.md`](results/intersectional.md).

## Interpretation

**Removing race.** Performance is essentially unchanged (accuracy 0.675 → 0.677,
F1 0.717 → 0.717), while the statistical parity difference shrinks slightly
(0.314 → 0.280). Masking the protected attribute narrows the gap at no cost in
performance here, but most of the disparity remains.

**Reweighing.** The four weights are:

| Group | Outcome | Weight |
|---|---|---|
| African-American | No recidivism (favourable) | 1.101 |
| African-American | Recidivism | 0.907 |
| Caucasian | No recidivism (favourable) | 0.877 |
| Caucasian | Recidivism | 1.188 |

Each weight is the expected share of a (race, outcome) cell, if race and outcome
were independent, divided by its observed share. A weight above 1 means the cell
is underrepresented relative to independence, so its instances count more in the
training loss; a weight below 1 means the opposite. Here, African-Americans who did
not reoffend and Caucasians who did are upweighted, which counteracts the
association between race and outcome in the training data.

The cost in performance is small: accuracy is 0.669 and F1 is 0.712, compared with
0.675 and 0.717 for the baseline. The fairness gain is much larger: the statistical
parity difference falls to 0.090, compared with 0.314 for the baseline, and the TPRs
of the two groups are now close (0.779 vs 0.761).

**Post-processing.** The best parameter set was chosen in two steps. First, an F1
floor was set at the baseline classifier's F1 minus 0.1 (an arbitrary tolerance);
then, among parameter sets above that floor, the one with the smallest
equalized-odds gap was selected. Without the floor, the search would likely pick
a parameter set with a very small gap but an F1 too low for the classifier to be useful.

The selected parameters are:

| Parameter | Original prediction | Race | P(new prediction = 1) |
|---|---|---|---|
| p_00 | 0 (no recidivism) | African-American | 0.145 |
| p_01 | 0 (no recidivism) | Caucasian | 0.136 |
| p_10 | 1 (recidivism) | African-American | 0.401 |
| p_11 | 1 (recidivism) | Caucasian | 0.767 |

Each parameter is the probability that the post-processed classifier predicts 1
(recidivism), given the original classifier's prediction and the person's race.
Favourable predictions are flipped to recidivism about equally often for both groups
(0.145 vs 0.136). The groups differ in how often a recidivism prediction is kept:
77% of the time for Caucasians, but only 40% of the time for African-Americans.
In other words, most of the correction comes from turning African-Americans'
recidivism predictions into favourable ones.

This brings the groups' TPRs (0.779 vs 0.786) and positive rates (0.720 vs 0.760)
close together, but it comes at the largest cost of all four classifiers: accuracy
falls to 0.563, precision to 0.564 and F1 to 0.656. Recall rises slightly (0.783 vs
0.774 for the baseline), because the classifier predicts the favourable outcome more
often overall. A gap in FPR remains (0.670 vs 0.716), and the selected parameters
had a smaller equalized-odds gap during the search (0.010) than on the fresh draw
reported here (0.053), which reflects the randomness of the method.

**Statistical parity and equal opportunity.** Strictly, no classifier satisfies
either criterion, since that would require exactly equal positive rates (statistical
parity) or exactly equal TPRs (equal opportunity). In practice some tolerance is
allowed, such as the 80% rule for statistical parity: the lower positive rate must be
at least 80% of the higher one. Under this rule, classifier 1 fails (0.485 / 0.799 = 0.607),
classifier 2 fails (0.492 / 0.772 = 0.637), classifier 3 passes (0.581 / 0.671 = 0.866)
and classifier 4 passes (0.720 / 0.760 = 0.947).

For equal opportunity, a common alternative to a ratio is a threshold on the absolute
TPR difference; here, 0.1. Under this rule, classifier 1 fails (|0.862 − 0.688| = 0.174),
classifier 2 fails (|0.844 − 0.695| = 0.149), classifier 3 passes (|0.761 − 0.779| = 0.018)
and classifier 4 passes (|0.786 − 0.779| = 0.007). This matches the verdicts under the
80% rule for statistical parity.

Overall, reweighing gives the best trade-off: it passes both criteria at almost no
cost in performance. Post-processing achieves the smallest gaps but loses the most
accuracy and makes predictions random for identical individuals.

**Intersectional fairness.** The disparities are driven mainly by the gap between
Caucasian women, who most often receive the favourable prediction (positive rate
0.904), and African-American men, who least often do (0.413). For classifier 1, the
maximum subgroup differences are 0.491 in positive rate, 0.347 in TPR and 0.511 in FPR.
Removing race barely changes this (0.474, 0.337 and 0.476).

## Limitations

- The post-processed classifier is randomized: identical individuals can receive
  different predictions, and its metrics vary from draw to draw. The parameter
  search uses a single draw per candidate, so the selected set partly reflects chance.
- A random search over parameters stands in for the linear program, so the
  selected parameters are not optimal.
- Results come from a single train/test split.
- The 80% rule (from US employment law) and the 0.1 TPR threshold are
  conventions, not statistical tests.
- `two_year_recid` records new arrests, not offences, so the labels themselves
  may reflect biased policing.

## Repository structure

```
compas_fairness/
    data.py            download, filtering, encoding, train/test split
    metrics.py         performance and group fairness metrics
    reweighing.py      reweighing weights
    postprocessing.py  equalized-odds parameter search and evaluation
    report.py          printed reports and results tables
run_experiments.py     runs all experiments and writes results/
tests/                 sanity checks on hand-made examples
results/               generated tables
```