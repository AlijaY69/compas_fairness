"""Reweighing: weight each (race, label) cell so that race and label are
independent in the weighted training data (Kamiran & Calders, 2012)."""


def compute_weights(X_train, y_train):
    """Return the four (race, label) weights and one weight per training row.

    w(race, label) = P(race) * P(label) / P(race, label)
    """
    train_data = X_train.copy()
    train_data["two_year_recid"] = y_train

    prob_dict = {
        "p_0": (X_train["race"] == 0).mean(),
        "p_1": (X_train["race"] == 1).mean(),
        "p__0": (y_train == 0).mean(),
        "p__1": (y_train == 1).mean(),
    }

    weight_dict = {}

    for race in (0, 1):
        for label in (0, 1):
            p_exp = prob_dict[f"p_{race}"] * prob_dict[f"p__{label}"]
            mask = (train_data["race"] == race) & (train_data["two_year_recid"] == label)
            p_obs = mask.mean()
            weight_dict[f"w_{race}{label}"] = p_exp / p_obs

    weights = []
    for (race, label) in zip(train_data["race"], train_data["two_year_recid"]):
        weights.append(weight_dict[f"w_{race}{label}"])

    return weight_dict, weights