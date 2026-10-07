"""Post-processing for equalized odds via a random search over the four
derived-predictor parameters p[(y_hat, race)] = P(y_tilde = 1 | y_hat, race)."""

import random

import pandas as pd

from compas_fairness.metrics import metrics_scores

N_SAMPLES = 5000
SEED = 88


def sample_parameters(n=N_SAMPLES, seed=SEED):
    """Step 1: draw n random parameter sets."""
    random.seed(seed)
    random_parameters = []
    for _ in range(n):
        p_00 = random.random()
        p_01 = random.random()
        p_10 = random.random()
        p_11 = random.random()
        random_parameters.append({(0, 0): p_00,
                                  (0, 1): p_01,
                                  (1, 0): p_10,
                                  (1, 1): p_11})
    return random_parameters


def build_post_data(X_test, y_pred, y_test):
    """Step 2: collect race, original predictions and true labels, plus the
    number of cases in each (prediction, race) condition."""
    df_post_data = pd.DataFrame({'race_num': X_test["race"],
                                 'pred_labels': y_pred,
                                 'true_labels': y_test})

    subset_sizes = {
        (0, 0): len(df_post_data.query('pred_labels == 0 & race_num == 0')),
        (0, 1): len(df_post_data.query('pred_labels == 0 & race_num == 1')),
        (1, 0): len(df_post_data.query('pred_labels == 1 & race_num == 0')),
        (1, 1): len(df_post_data.query('pred_labels == 1 & race_num == 1')),
    }
    return df_post_data, subset_sizes


def generate_labels(subset_sizes, p_dict):
    """
    subset_sizes: dict with number of cases falling in each condition
    p_dict: the postprocessing parameters
    """
    new_predictions = {}

    for (prediction, group), p in p_dict.items():
        # The number of instances for which we need to generate labels
        num_instances = subset_sizes[(prediction, group)]

        new_labels = [1 if random.random() < p else 0 for _ in range(num_instances)]

        # save the new predictions
        new_predictions[(prediction, group)] = new_labels

    return new_predictions


def apply_labels(df_post_data, new_predictions):
    """Return a copy of df_post_data with the new labels written back."""
    df_copy = df_post_data.copy()
    for (pred, group), new_preds in new_predictions.items():
        mask = (df_post_data["pred_labels"] == pred) & (df_post_data["race_num"] == group)
        df_copy.loc[mask, "pred_labels"] = new_preds
    return df_copy


def search_parameters(df_post_data, subset_sizes, X_test, random_parameters, seed=SEED):
    """Step 3: apply every parameter set and store its metrics."""
    random.seed(seed)
    metrics = []
    for p_dict in random_parameters:
        new_predictions = generate_labels(subset_sizes, p_dict)
        df_copy = apply_labels(df_post_data, new_predictions)
        res = metrics_scores(df_copy["true_labels"], df_copy["pred_labels"], X_test)
        metrics.append([res, p_dict])
    return metrics


def select_best(metrics, f1_floor):
    """Step 4: smallest equalized-odds gap among sets with F1 above f1_floor.

    Gap = |TPR_CC - TPR_AA| + |FPR_CC - FPR_AA|.
    """
    gap = float("inf")
    best_params = {"params": None, "gap": gap, "f1": 0}

    for res, p_dict in metrics:
        gm = res["group_metrics"]
        eo_gap = abs(gm["1_tpr"] - gm["0_tpr"]) + abs(gm["1_fpr"] - gm["0_fpr"])
        if res["f1"] > f1_floor and eo_gap < gap:
            gap = eo_gap
            best_params = {"params": p_dict, "gap": eo_gap, "f1": res["f1"]}

    if best_params["params"] is None:
        raise ValueError(f"No parameter set reached F1 > {f1_floor:.3f}")
    return best_params


def evaluate_best(df_post_data, subset_sizes, X_test, best_p, seed=SEED):
    """Step 5: regenerate labels with the chosen parameters and evaluate them."""
    random.seed(seed)
    new_predictions = generate_labels(subset_sizes, best_p)
    df_final = apply_labels(df_post_data, new_predictions)
    return metrics_scores(df_final["true_labels"], df_final["pred_labels"], X_test)