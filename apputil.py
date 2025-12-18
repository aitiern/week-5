# apputil.py
import pandas as pd
import plotly.express as px


# ---------- Exercise 1: Survival Patterns ----------

def survival_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group Titanic passengers by Pclass, Sex, and age_group and compute:
      n_passengers, n_survivors, survival_rate

    Returns all combinations of (Pclass, Sex, age_group), including those with 0 members.
    Ensures age_group is pandas Categorical dtype.
    """
    df = df.copy()

    # Age groups as categorical
    labels = ["Child (0–12)", "Teen (13–19)", "Adult (20–59)", "Senior (60+)"]
    bins = [-1, 12, 19, 59, 120]
    age_cat = pd.CategoricalDtype(categories=labels, ordered=True)

    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True).astype(age_cat)

    # Aggregate observed groups
    grouped = (
        df.dropna(subset=["age_group"])
          .groupby(["Pclass", "Sex", "age_group"], observed=False)
          .agg(
              n_passengers=("Survived", "size"),
              n_survivors=("Survived", "sum")
          )
    )

    # Force all combinations (including empty groups)
    all_pclass = sorted(df["Pclass"].dropna().unique().tolist())
    all_sex = sorted(df["Sex"].dropna().unique().tolist())
    full_index = pd.MultiIndex.from_product(
        [all_pclass, all_sex, labels],
        names=["Pclass", "Sex", "age_group"]
    )

    grouped = grouped.reindex(full_index, fill_value=0).reset_index()

    # Keep categorical dtype after reindex/reset
    grouped["age_group"] = grouped["age_group"].astype(age_cat)

    # Survival rate (avoid divide-by-zero)
    grouped["survival_rate"] = grouped.apply(
        lambda r: (r["n_survivors"] / r["n_passengers"]) if r["n_passengers"] > 0 else 0.0,
        axis=1
    )

    # Easy-to-read ordering
    grouped = grouped.sort_values(["Pclass", "Sex", "age_group"]).reset_index(drop=True)

    return grouped


def visualize_demographic(summary: pd.DataFrame):
    """
    Plot survival_rate by age_group, split by Sex and faceted by Pclass.
    """
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


# ---------- Exercise 2: Family Size and Wealth ----------

def family_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create family_size = SibSp + Parch + 1.
    Group by (family_size, Pclass) and compute:
      n_passengers, avg_fare, min_fare, max_fare
    Returns a sorted table for interpretability.
    """
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
          .sort_values(["Pclass", "family_size"])
          .reset_index(drop=True)
    )
    return grouped


def last_names(df: pd.DataFrame) -> pd.Series:
    """
    Extract last names from Name column ("Last, Title. First") and return value counts.
    """
    last = df["Name"].astype(str).str.split(",", n=1).str[0].str.strip()
    return last.value_counts()


def visualize_families(summary: pd.DataFrame):
    """
    Plot avg_fare vs family_size, colored by Pclass.
    """
    fig = px.line(
        summary,
        x="family_size",
        y="avg_fare",
        color="Pclass",
        markers=True,
        hover_data=["min_fare", "max_fare", "n_passengers"]
    )
    return fig


# ---------- Bonus: Age Division by Class Median ----------

def determine_age_division(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add older_passenger:
      True  if Age > median(Age) within passenger's class
      False if Age <= median(Age) within class
      None  if Age is missing
    """
    df = df.copy()
    class_medians = df.groupby("Pclass")["Age"].median()

    df["older_passenger"] = df.apply(
        lambda row: (row["Age"] > class_medians[row["Pclass"]]) if pd.notnull(row["Age"]) else None,
        axis=1
    )
    return df


def visualize_age_division(df: pd.DataFrame):
    """
    Plot survival_rate by Sex and older_passenger, faceted by Pclass.
    """
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


# --- END OF FILE ---
