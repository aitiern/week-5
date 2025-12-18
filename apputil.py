# apputil.py
import pandas as pd
import plotly.express as px

# ---------- Exercise 1: Survival Patterns ----------

def survival_demographics(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 12, 19, 59, 120]
    labels = ["Child (0–12)", "Teen (13–19)", "Adult (20–59)", "Senior (60+)"]

    df = df.copy()

    # Force categorical dtype
    cat_type = pd.CategoricalDtype(categories=labels, ordered=True)
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels, include_lowest=True).astype(cat_type)

    # Aggregate observed groups
    g = (
        df.dropna(subset=["age_group"])
          .groupby(["Pclass", "Sex", "age_group"], observed=False)
          .agg(
              n_passengers=("Survived", "size"),
              n_survivors=("Survived", "sum"),
          )
    )

    # Reindex to include ALL combinations (including those with 0 members)
    all_pclass = sorted(df["Pclass"].dropna().unique().tolist())
    all_sex = sorted(df["Sex"].dropna().unique().tolist())
    all_age = list(cat_type.categories)

    full_index = pd.MultiIndex.from_product(
        [all_pclass, all_sex, all_age],
        names=["Pclass", "Sex", "age_group"]
    )

    g = g.reindex(full_index, fill_value=0).reset_index()

    # Ensure categorical dtype survives reindex/reset
    g["age_group"] = g["age_group"].astype(cat_type)

    # survival_rate with safe divide
    g["survival_rate"] = g.apply(
        lambda r: (r["n_survivors"] / r["n_passengers"]) if r["n_passengers"] > 0 else 0.0,
        axis=1
    )

    return g


def visualize_demographic(summary: pd.DataFrame):
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
    df = df.copy()
    df["family_size"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0) + 1

    grouped = (
        df.groupby(["family_size", "Pclass"])
          .agg(
              n_passengers=("Fare", "size"),
              avg_fare=("Fare", "mean"),
              min_fare=("Fare", "min"),
              max_fare=("Fare", "max"),
          )
          .reset_index()
    )
    return grouped


def last_names(df: pd.DataFrame) -> pd.Series:
    last = df["Name"].astype(str).str.split(",", n=1).str[0].str.strip()
    return last.value_counts()


def visualize_families(summary: pd.DataFrame):
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
    df = df.copy()
    class_medians = df.groupby("Pclass")["Age"].median()

    df["older_passenger"] = df.apply(
        lambda row: (row["Age"] > class_medians[row["Pclass"]]) if pd.notnull(row["Age"]) else None,
        axis=1
    )
    return df


def visualize_age_division(df: pd.DataFrame):
    grouped = (
        df.dropna(subset=["older_passenger"])
          .groupby(["Pclass", "Sex", "older_passenger"])
          .agg(
              n_passengers=("Survived", "size"),
              n_survivors=("Survived", "sum"),
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
