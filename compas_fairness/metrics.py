"""Compute fairness and performance metrics."""

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def metrics_scores(y_true, y_pred, X_test):  
    """Return performance and group fairness metrics.

    0 (non-recidivism) is treated as the positive outcome/label.
    Groups: "0" = African-American, "1" = Caucasian; intersectional
    groups are race then sex (0 = female, 1 = male), e.g. "01" = AA male.
    """
    f1 = f1_score(y_true, y_pred, pos_label=0)
    precision = precision_score(y_true, y_pred, pos_label=0)
    recall = recall_score(y_true, y_pred, pos_label=0)
    accuracy = accuracy_score(y_true, y_pred)

    race_sex_list = []
    for race, sex in zip(X_test["race"], X_test["sex"]):
        race_sex_list.append(f"{race}{sex}")

    groups = ("0", "1", "00", "10", "01", "11")
    metrics = ("tp", "fp", "tn", "fn", "tpr", "fpr", "preds", "sp")

    group_metrics = {f"{group}_{metric}": 0 for group in groups for metric in metrics}

    # calculate number of TP, FP, TN and FN 
    for rs, y1, y2 in zip(race_sex_list, y_true, y_pred):
        if y1 == y2 == 0:
            metric = "tp"
            group_metrics[f"{rs}_preds"] += 1     # increment the subgroup (race x sex)
            group_metrics[f"{rs[0]}_preds"] += 1  # increment the race group
        elif y1 == y2 == 1:
            metric = "tn"
        elif y1 == 1:
            metric = "fp"
            group_metrics[f"{rs}_preds"] += 1
            group_metrics[f"{rs[0]}_preds"] += 1
        else:
            metric = "fn"
        group_metrics[f"{rs}_{metric}"] += 1
        group_metrics[f"{rs[0]}_{metric}"] += 1

    # calculate FPR and TPR using the values calculated above
    for g in groups:
        for rate in ("fpr", "tpr"):
            this = "tp" if rate == "tpr" else "fp"
            other = "fn" if rate == "tpr" else "tn"
            group_metrics[f"{g}_{rate}"] = group_metrics[f"{g}_{this}"]/(group_metrics[f"{g}_{this}"] + group_metrics[f"{g}_{other}"])

    # calculate statistical parity for each group
    for g in groups:
        n = sum(group_metrics[f"{g}_{m}"] for m in ("tp", "fp", "tn", "fn"))
        group_metrics[f"{g}_sp"] = group_metrics[f"{g}_preds"] / n

    sps, tprs, fprs = [], [], []
    for g in ("00", "10", "01", "11"):
        sps.append(group_metrics[f"{g}_sp"])
        tprs.append(group_metrics[f"{g}_tpr"])
        fprs.append(group_metrics[f"{g}_fpr"])

    # calculate maximum differences between subgroups (order: SP, TPR, FPR)
    maxmins = []
    for group in [sps, tprs, fprs]:
        maxval = max(group)
        minval = min(group)
        maxmin = maxval-minval
        maxmins.append(maxmin)

    return { 
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "group_metrics": group_metrics,
        "maxmins": maxmins,
    }