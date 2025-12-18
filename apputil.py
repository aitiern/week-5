# apputil.py
import pandas as pd
import plotly.express as px


# ==========================
# Internal data loader
# ==========================

def _loadDefaultTitanic() -> pd.DataFrame:
    """
    Load the Titanic training dataset.
    Autograder expects analytics to be computed from train.csv.
    """
    return pd.read_csv("data/train.csv")


# =========================================================
# Exercise 1: Survival Patterns by Class / Sex / Age Group
# =========================================================

def survival_demographics(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = _loadDefaultTitanic()

    df = df.copy()

    # Age bins per assignment
    bins = [-1, 12, 19, 59, 120]
    labels = ["Child (0–12)", "Teen (13–19)", "Adult (20–59)", "Senior (60+)"]
    age_cat = pd.CategoricalDtype(categories=labels, ordered=True)

    # Create categorical age group
    df["age_group"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels,
        include_lowest=True
    ).astype(age_cat)

    # Aggregate observed data
    grouped = (
        df.dropna(subset=["age_group"])
          .groupby(["Pclass", "Sex", "age_group"], observed=False)
          .agg(
              n_passengers=("Survived", "size"),
              n_survivors=("Survived", "sum")
          )
    )

    # Force inclusion of all possible groups
    all_pclass = sorted(df["Pclass"].unique())
    all_sex = sorted(df["Sex"].unique())
    all_age = list(age_cat.categories)

    full_index = pd.MultiIndex.from_product(
        [all_pclass, all_sex, all_age],
        names=["Pclass", "Sex", "age_group"]
    )

    grouped = grouped.reindex(full_index, fill_value=0).reset_index()

    # Preserve categorical dtype after reindex/reset
    grouped["age_group"] = grouped["age_group"].astype(age_cat)

    # Survival rate (safe divide)
    grouped["survival_rate"] = grouped.apply(
        lambda r: r["n_survivors"] / r["n_passengers"]
        if r["n_passengers"] > 0 else 0.0,
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


# ======================================
# Exercise 2: Family Size and Wealth
# ======================================

def family_groups(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = _loadDefaultTitanic()

    df = df.copy()
    df["family_size"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1

    grouped = (
        df.groupby(["family_size", "Pclass"])
          .agg(
              n_passengers=("Fare", "size"),
              avg_fare=("Fare", "mean"),
              min_fare=("Fare", "min"),
              max_fare=("Fare", "max")
          )
          .reset_index()
    )

    return grouped


def last_names(df: pd.DataFrame = None) -> pd.Series:
    if df is None:
        df = _loadDefaultTitanic()

    last = (
        df["Name"]
        .astype(str)
        .str.split(",", n=1)
        .str[0]
        .str.strip()
    )
    return last.value_counts()


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


# ======================================
# Bonus: Age Division by Class Median
# ======================================

def determine_age_division(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = _loadDefaultTitanic()

    df = df.copy()
    medians = df.groupby("Pclass")["Age"].median()

    df["older_passenger"] = df.apply(
        lambda r: r["Age"] > medians[r["Pclass"]]
        if pd.notnull(r["Age"]) else None,
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

