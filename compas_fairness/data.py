"""Load, filter, encode and split the ProPublica COMPAS two-year recidivism data."""

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = ("https://raw.githubusercontent.com/propublica/compas-analysis/"
            "master/compas-scores-two-years.csv")
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "compas-scores-two-years.csv"

SEED = 88
TEST_SIZE = 0.2

# Columns related to COMPAS scores or otherwise unused (as in the original assignment)
SCORE_COLUMNS = [
    "is_violent_recid", "violent_recid", "is_recid", "name", "first", "last",
    "compas_screening_date", "decile_score", "days_b_screening_arrest",
    "c_days_from_compas", "type_of_assessment", "score_text", "screening_date",
    "in_custody", "out_custody", "start", "end", "event", "c_arrest_date",
]

# Explicit encodings (same codes LabelEncoder produced, but self-documenting)
ENCODINGS = {
    "race": {"African-American": 0, "Caucasian": 1},
    "sex": {"Female": 0, "Male": 1},
    "age_cat": {"Less than 25": 0, "25 - 45": 1, "Greater than 45": 2},
    "c_charge_degree": {"F": 0, "M": 1},
}

# Columns excluded from the feature matrix
NON_FEATURES = [
    "two_year_recid", "id", "age", "dob", "c_jail_in", "c_jail_out",
    "c_case_number", "c_charge_desc", "c_offense_date",
]


def download(path: Path = DATA_PATH, url: str = DATA_URL) -> Path:
    """Download the CSV once; reuse the local copy afterwards."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urlretrieve(url, path)
    return path


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the raw data and apply ProPublica's filters."""
    df = pd.read_csv(download(path))

    df = df[(df.days_b_screening_arrest <= 30)
            & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1)
            & (df.c_charge_degree != "O")
            & (df.score_text != "N/A")
            & (df.race.isin(["Caucasian", "African-American"]))]

    df = df.loc[:, ~(df.columns.str.endswith(".1")          # duplicate columns
                     | df.columns.str.startswith(("r_", "v_", "vr_"))
                     | df.columns.isin(SCORE_COLUMNS))]
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical columns and derive sentence length in days."""
    df = df.copy()
    for col, mapping in ENCODINGS.items():
        df[col] = df[col].map(mapping)
        if df[col].isna().any():
            raise ValueError(f"Unmapped values in column '{col}'")
    df["sentence_length"] = (pd.to_datetime(df["c_jail_out"])
                             - pd.to_datetime(df["c_jail_in"])).dt.days
    return df


def split(df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = SEED):
    """Return X_train, X_test, y_train, y_test."""
    y = df["two_year_recid"]
    X = df.drop(columns=NON_FEATURES)
    return train_test_split(X, y, test_size=test_size, random_state=seed)