# Fairness-Aware Recidivism Classification on COMPAS

**Machine Learning and Data Mining (Module 2), A.Y. 2024/25 — Final Assignment**
University of Bologna · Kosar Mazaheri

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kosaremazahery/REPO-NAME/blob/main/notebooks/compas_fairness.ipynb)

---

## The question

COMPAS is a risk assessment tool used in US courts to predict whether a defendant will re-offend. In 2016 ProPublica reported that among defendants who never re-offended, Black defendants were flagged as high risk about twice as often as white ones. The vendor replied that the tool is calibrated within groups. Both claims turned out to be true, and later work showed the two criteria cannot both hold once base rates differ.

Rather than relitigate that dispute, this project asks something narrower and testable:

> If I train an ordinary classifier on the same data, and never show it COMPAS's scores, does it come out biased anyway?

If the answer is yes, the bias is a property of the data rather than of COMPAS's proprietary algorithm, and building a better model will not fix it.

## What the notebook does

1. Cleans the raw ProPublica data with their four published filters (7,214 → 6,172 defendants).
2. Selects five screening-time features, excluding the other 48 columns for leakage, circularity, or lack of signal.
3. Compares logistic regression, a decision tree, and a random forest by 5-fold cross-validation, then tunes the forest by grid search.
4. Audits the resulting model for fairness across racial groups, and audits COMPAS's own scores on the same defendants for comparison.
5. Tests the most intuitive mitigation (removing `race` from the feature set) and measures what it actually achieves.

## Findings

**Five features match a 137-item questionnaire.** The tuned random forest reaches 0.684 accuracy against a naive baseline of 0.545 and COMPAS's own 0.661 on the same test defendants. Every model family tested lands between 0.67 and 0.69, which suggests the ceiling comes from the data rather than from model capacity.

**Comparing fairness requires matching the selection rate.** At scikit-learn's default 0.5 threshold my model flagged 37.3% of defendants while COMPAS flagged 44.2%, and its false positive rates were lower for *both* racial groups. That looks like fairness and is not: a predictor that flags fewer people has a lower false positive rate everywhere, automatically. Lowering the threshold to 0.461 equalises the selection rates, and at that point the conclusion reverses. My model imposes an identical false-positive burden on Black defendants (0.411, exactly COMPAS's figure) while giving white defendants more benefit of the doubt (0.188 against 0.227). The disparity ratio rises from 1.81 to 2.19.

**The mechanism is proxy encoding.** `priors_count` is the dominant feature (importance 0.50) and is heavily race-correlated: Black defendants are 51% of the sample but 75% of the high-priors tail. Racial information therefore reaches the model whether or not `race` is supplied.

**Fairness through unawareness fails.** Removing `race` and retraining leaves the false positive rate for Black defendants unchanged (0.337 to 0.339). The gap narrows only because the rate for white defendants rises (0.133 to 0.173). The intervention redistributes harm without reducing it.

**Two things the aggregate metrics hide.** The group flagged most often (Black men with felony charges, 62.7%) is not the group flagged most unjustly (Black women with felony charges, over-flagged by 14.7 points). And the model introduced a disparity with respect to sex roughly four times larger than COMPAS's own, which no aggregate racial metric would have revealed.



## Running it

Click the Colab badge above and choose Runtime → Run all. Nothing needs downloading: the notebook reads the data straight from ProPublica's repository, and every random seed is fixed (`RANDOM_STATE = 42`), so the numbers reproduce exactly.

To run it locally instead:

```bash
pip install -r requirements.txt
jupyter notebook notebooks/compas_fairness.ipynb
```

## Data

`compas-scores-two-years.csv` from [propublica/compas-analysis](https://github.com/propublica/compas-analysis), covering roughly 7,000 pretrial defendants screened in Broward County, Florida.

**A caveat that bounds everything above.** `priors_count` and the charge records measure *arrests*, which reflect policing and enforcement patterns as much as offending behaviour. The target label inherits the same defect, since re-arrest is only a proxy for re-offence. Every disparity reported here describes the criminal-justice record rather than the people in it.

## References

- Angwin, Larson, Mattu, Kirchner (2016). *Machine Bias.* ProPublica.
- Chouldechova (2017). *Fair prediction with disparate impact.* Big Data 5(2).
- Kleinberg, Mullainathan, Raghavan (2017). *Inherent trade-offs in the fair determination of risk scores.* ITCS.
- Kamiran, Calders (2012). *Data preprocessing techniques for classification without discrimination.* KAIS 33(1).
- Hardt, Price, Srebro (2016). *Equality of opportunity in supervised learning.* NIPS.
- Dressel, Farid (2018). *The accuracy, fairness, and limits of predicting recidivism.* Science Advances 4(1).
