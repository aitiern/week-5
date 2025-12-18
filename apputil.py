# apputil.py
import pandas as pd
import plotly.express as px
from pathlib import Path


# ==========================
# Internal data loader
# ==========================

def _loadDefaultTitanic() -> pd.DataFrame:
    base = Path(__file__).resolve().parent

    candidates = [
        base / "data" / "train.csv",
        base / "train.csv",
        Path.cwd() / "data" / "train.csv",
        Path.cwd() / "train.csv",
    ]

    for p in candidates:
        if p.exists():
            return pd.read_csv(p)

    raise FileNotFoundError("Could not locate train.csv")


# ---------- Exercise 1 ----------

def survival_demographics(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = _loadDefaultTitanic()

    labels = ["Child (0–12)", "Teen (13–19)", "Adult (20–59)", "Senior (60+)"]
    bins = [-1, 12, 19, 59, 120]
    cat = pd.CategoricalDtype(categories=labels, ordered=True)

    df = df.copy()
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True).astype(cat)

    grouped = (
        df.dropna(subset=["age_group"])
          .groupby(["Pclass", "Sex", "age_group"], observed=False)
          .agg(
              n_passengers=("Survived", "size"),
              n_survivors=("Survived", "sum")
          )
    )

    full_index = pd.MultiIndex.from_product(
        [sorted(df["Pclass"].unique()), sorted(df["Sex"].unique()), labels],
        names=["Pclass", "Sex", "age_group"]
    )

    grouped = grouped.reindex(full_index, fill_value=0).reset_index()
    grouped["age_group"] = grouped["age_group"].astype(cat)

    grouped["survival_rate"] = grouped.apply(
        lambda r: r["n_survivors"] / r["n_passengers"] if r["n_passengers"] > 0 else 0.0,
        axis=1
    )

    return grouped


def visualize_demographic(summary: pd.DataFrame = None):
    if summary is None:
        summary = survival_demographics()

    fig = px.bar(
        summary,
        x="age_group",
        y="survival_rate",
        color="Sex",
        facet_col="Pclass",
        barmode="group",
        text=summary["survival_rate"].apply(lambda x: f"{x:.0%}")
    )
    fig.update_yaxes(tickformat=".0%")
    return fig


# ---------- Exercise 2 ----------

def family_groups(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = _loadDefaultTitanic()

    df = df.copy()
    df["family_size"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1

    return (
        df.groupby(["family_size", "Pclass"])
          .agg(
              n_passengers=("Fare", "size"),
              avg_fare=("Fare", "mean"),
              min_fare=("Fare", "min"),
              max_fare=("Fare", "max")
          )
          .reset_index()
    )


def last_names(df: pd.DataFrame = None) -> pd.Series:
    if df is None:
        df = _loadDefaultTitanic()

    return df["Name"].str.split(",", n=1).str[0].str.strip().value_counts()


def visualize_families(summary: pd.DataFrame = None):
    if summary is None:
        summary = family_groups()

    fig = px.line(
        summary,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True,
        hover_data=["min_fare", "max_fare", "n_passengers"]
    )
    return fig


# ---------- Bonus ----------

def determine_age_division(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = _loadDefaultTitanic()

    medians = df.groupby("Pclass")["Age"].median()
    df = df.copy()

    df["older_passenger"] = df.apply(
        lambda r: r["Age"] > medians[r["Pclass"]] if pd.notnull(r["Age"]) else None,
        axis=1
    )

    return df


def visualize_age_division(df: pd.DataFrame = None):
    if df is None:
        df = determine_age_division()

    grouped = (
        df.dropna(subset=["older_passenger"])
          .groupby(["Pclass", "Sex", "older_passenger"])
          .agg(
              n_passengers=("Survived", "size"),
              n_survivors=("Survived", "sum")
          )
          .reset_index()
    )

    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]

    fig = px.bar(
        grouped,
        x="Sex",
        y="survival_rate",
        color="older_passenger",
        facet_col="Pclass",
        barmode="group",
        text=grouped["survival_rate"].apply(lambda x: f"{x:.0%}")
    )
    fig.update_yaxes(tickformat=".0%")
    return fig
