# app.py
import streamlit as st
import pandas as pd
from apputil import (
    survivalDemographics, visualizeDemographic,
    familyGroups, visualizeFamilies,
    lastNames,
    determineAgeDivision, visualizeAgeDivision,
)

st.set_page_config(page_title="Titanic Analysis", layout="wide")

# --- Load data ---
# Place your titanic CSV in the /data folder of your repo
@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

df = load_data()

st.title("🚢 Titanic Data Exploration")

# ---------- Exercise 1 ----------
st.header("Exercise 1: Survival Patterns")
surv_tbl = survivalDemographics(df)
st.write("**Question:** Within each class, did females have higher survival rates than males across age groups?")
st.dataframe(surv_tbl)
st.plotly_chart(visualizeDemographic(surv_tbl), use_container_width=True)

# ---------- Exercise 2 ----------
st.header("Exercise 2: Family Size and Wealth")
fam_tbl = familyGroups(df)
st.dataframe(fam_tbl)
st.plotly_chart(visualizeFamilies(fam_tbl), use_container_width=True)

st.subheader("Most Common Last Names")
name_counts = lastNames(df)
st.write(name_counts.head(20))

# ---------- Bonus ----------
st.header("Bonus: Age Division vs Survival")
df2 = determineAgeDivision(df)
st.plotly_chart(visualizeAgeDivision(df2), use_container_width=True)
