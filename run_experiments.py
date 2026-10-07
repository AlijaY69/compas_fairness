"""Run all four classifiers and the intersectional analysis; write tables to results/."""

from pathlib import Path

from sklearn.linear_model import LogisticRegression

from compas_fairness.data import SEED, encode, load_data, split
from compas_fairness.metrics import metrics_scores
from compas_fairness.postprocessing import (build_post_data, evaluate_best, sample_parameters,
                                            search_parameters, select_best)
from compas_fairness.report import print_report, print_weights, write_intersectional, write_summary
from compas_fairness.reweighing import compute_weights

X_train, X_test, y_train, y_test = split(encode(load_data()))

# 5.1 Regular classification
lr_1 = LogisticRegression(random_state=SEED).fit(X_train, y_train)
y_pred_1 = lr_1.predict(X_test)
res_1 = metrics_scores(y_test, y_pred_1, X_test)
print_report(res_1, intersect=True)

# 5.2 Without the protected attribute
lr_2 = LogisticRegression(random_state=SEED).fit(X_train.drop("race", axis=1), y_train)
y_pred_2 = lr_2.predict(X_test.drop("race", axis=1))
res_2 = metrics_scores(y_test, y_pred_2, X_test)
print_report(res_2, intersect=True)

# 5.3 Reweighing
weight_dict, weights = compute_weights(X_train, y_train)
print_weights(weight_dict)
lr_3 = LogisticRegression(random_state=SEED).fit(X_train, y_train, sample_weight=weights)
y_pred_3 = lr_3.predict(X_test)
res_3 = metrics_scores(y_test, y_pred_3, X_test)
print_report(res_3)

# 5.4 Post-processing for equalized odds (on classifier 1's predictions)
df_post_data, subset_sizes = build_post_data(X_test, y_pred_1, y_test)
candidates = search_parameters(df_post_data, subset_sizes, X_test, sample_parameters())
best_params = select_best(candidates, f1_floor=res_1["f1"] - 0.1)
print(best_params)
res_4 = evaluate_best(df_post_data, subset_sizes, X_test, best_params["params"])
print_report(res_4)

# Results tables
RESULTS_DIR = Path(__file__).resolve().parent / "results"
write_summary({"1. Logistic regression": res_1,
               "2. Without race": res_2,
               "3. Reweighing": res_3,
               "4. Equalized-odds post-processing": res_4},
              RESULTS_DIR / "summary.md")
# Section 6: intersectional analysis for classifiers 1 and 2
write_intersectional({"1. Logistic regression": res_1, "2. Without race": res_2},
                     RESULTS_DIR / "intersectional.md")