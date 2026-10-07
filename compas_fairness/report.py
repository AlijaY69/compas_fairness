"""Print performance and fairness results returned by metrics_scores."""


def print_report(res, intersect=False):
    """Print overall and race-level metrics; add intersectional ones if intersect=True."""
    accuracy = res["accuracy"]
    precision = res["precision"]
    recall = res["recall"]
    f1 = res["f1"]
    group_metrics = res["group_metrics"]
    maxmins = res["maxmins"]

    print(
        f"""
        Performance Metrics
        Accuracy: {round(accuracy, 3)}
        Precision: {round(precision, 3)}
        Recall: {round(recall, 3)}
        F1 Score: {round(f1, 3)}

        Statistical Parity
        Statistical Parity African-Americans: {round(group_metrics["0_sp"], 3)}
        Statistical Parity Caucasians: {round(group_metrics["1_sp"], 3)}
        Statistical Parity Difference (CC - AA): {round(group_metrics["1_sp"] - group_metrics["0_sp"], 3)}

        TPR/FPR
        African-Americans
        TPR: {round(group_metrics["0_tpr"], 3)}
        FPR: {round(group_metrics["0_fpr"], 3)}

        Caucasians
        TPR: {round(group_metrics["1_tpr"], 3)}
        FPR: {round(group_metrics["1_fpr"], 3)}
        """
    )

    if intersect:
        print(
            f"""
        Intersectional Fairness

        TPR
        TPR Female African-Americans: {round(group_metrics["00_tpr"], 3)}
        TPR Female Caucasians: {round(group_metrics["10_tpr"], 3)}
        TPR Male African-Americans: {round(group_metrics["01_tpr"], 3)}
        TPR Male Caucasians: {round(group_metrics["11_tpr"], 3)}
        Maximum TPR Difference: {round(maxmins[1], 3)}

        FPR
        FPR Female African-Americans: {round(group_metrics["00_fpr"], 3)}
        FPR Female Caucasians: {round(group_metrics["10_fpr"], 3)}
        FPR Male African-Americans: {round(group_metrics["01_fpr"], 3)}
        FPR Male Caucasians: {round(group_metrics["11_fpr"], 3)}
        Maximum FPR Difference: {round(maxmins[2], 3)}

        Statistical Parity
        Statistical Parity Female African-Americans: {round(group_metrics["00_sp"], 3)}
        Statistical Parity Female Caucasians: {round(group_metrics["10_sp"], 3)}
        Statistical Parity Male African-Americans: {round(group_metrics["01_sp"], 3)}
        Statistical Parity Male Caucasians: {round(group_metrics["11_sp"], 3)}
        Maximum Statistical Parity Difference: {round(maxmins[0], 3)}
        """
        )


def print_weights(weight_dict):
    """Print the four reweighing weights."""
    for key, w in weight_dict.items():
        print(f"{key}: {w:.3f}")


def write_summary(results, path):
    """Write a markdown table comparing classifiers on performance and race-level fairness.

    results: dict mapping classifier name -> dict returned by metrics_scores
    """
    lines = [
        "| Classifier | Accuracy | Precision | Recall | F1 | SP AA | SP CC | SP diff (CC - AA) "
        "| TPR AA | TPR CC | FPR AA | FPR CC |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, res in results.items():
        gm = res["group_metrics"]
        row = [res["accuracy"], res["precision"], res["recall"], res["f1"],
               gm["0_sp"], gm["1_sp"], gm["1_sp"] - gm["0_sp"],
               gm["0_tpr"], gm["1_tpr"], gm["0_fpr"], gm["1_fpr"]]
        lines.append(f"| {name} | " + " | ".join(f"{v:.3f}" for v in row) + " |")
    _write(path, "Performance and race-level fairness", lines)


def write_intersectional(results, path):
    """Write markdown tables of race x sex subgroup metrics and their maximum differences.

    results: dict mapping classifier name -> dict returned by metrics_scores
    """
    labels = {"00": "AA female", "10": "CC female", "01": "AA male", "11": "CC male"}
    lines = []
    for name, res in results.items():
        gm = res["group_metrics"]
        lines += [f"### {name}", "",
                  "| Subgroup | SP | TPR | FPR |",
                  "|---|---|---|---|"]
        for g, label in labels.items():
            lines.append(f"| {label} | {gm[f'{g}_sp']:.3f} | {gm[f'{g}_tpr']:.3f} | {gm[f'{g}_fpr']:.3f} |")
        sp_gap, tpr_gap, fpr_gap = res["maxmins"]
        lines += [f"| **Max difference** | {sp_gap:.3f} | {tpr_gap:.3f} | {fpr_gap:.3f} |", ""]
    _write(path, "Intersectional fairness (race x sex)", lines)


def _write(path, title, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    note = ("Positive class: 0 (no recidivism). SP = share predicted favourable; "
            "TPR/FPR computed with 0 as the positive class.")
    path.write_text("\n".join([f"## {title}", "", note, ""] + lines) + "\n")