# apputil.py
import pandas as pd
import plotly.express as px


# ---------- Exercise 1: Survival Patterns ----------

def survivalDemographics(df: pd.DataFrame) -> pd.DataFrame:
    bins = [-1, 12, 19, 59, 120]
    labels = ["Child (0–12)", "Teen (13–19)", "Adult (20–59)", "Senior (60+)"]
    df = df.copy()
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)

    grouped = (
        df.dropna(subset=["age_group"])
          .groupby(["Pclass", "Sex", "age_group"])
          .agg(
              n_passengers=("Survived", "size"),
              n_survivors=("Survived", "sum")
          )
          .reset_index()
    )
    grouped["survival_rate"] = grouped["n_survivors"] / grouped["n_passengers"]
    return grouped


def visualizeDemographic(summary: pd.DataFrame):
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

def familyGroups(df: pd.DataFrame) -> pd.DataFrame:
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


def lastNames(df: pd.DataFrame) -> pd.Series:
    last = df["Name"].astype(str).str.split(",", n=1).str[0].str.strip()
    return last.value_counts()


def visualizeFamilies(summary: pd.DataFrame):
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

def determineAgeDivision(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    class_medians = df.groupby("Pclass")["Age"].median()
    df["older_passenger"] = df.apply(
        lambda row: row["Age"] > class_medians[row["Pclass"]]
        if pd.notnull(row["Age"]) else None,
        axis=1
    )
    return df


def visualizeAgeDivision(df: pd.DataFrame):
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