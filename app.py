import streamlit as st
import pandas as pd
from datetime import date
import tempfile, os

from apra_core import (
    analyze_project,
    monte_carlo_project_duration,
    summarize_samples,
    make_task_risk_fig,
    make_monte_carlo_fig,
)

# =========================
# Helpers
# =========================

def analyze_project_from_df(df_in, today):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        df_in.to_csv(tmp.name, index=False)
        path = tmp.name
    try:
        return analyze_project(path, today=today)
    finally:
        os.remove(path)


def validate_csv(df):
    required = ["Task", "Start", "Due", "Progress"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()


# =========================
# UI
# =========================

st.set_page_config(page_title="APRA Dashboard", layout="wide")
st.title("APRA — Project Risk Dashboard")

with st.sidebar:
    uploaded = st.file_uploader("Upload tasks CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data", value=False)
    sims = st.slider("Monte Carlo simulations", 500, 20000, 5000, step=500)
    today = st.date_input("Today", value=date.today())

    with open("sample_tasks.csv", "rb") as f:
        st.download_button(
            "Download sample CSV",
            f,
            file_name="sample_tasks.csv",
            mime="text/csv",
        )

# =========================
# Data load (single path)
# =========================

if use_demo:
    df_in = pd.read_csv("sample_tasks.csv")
elif uploaded:
    df_in = pd.read_csv(uploaded)
else:
    st.info("Using demo data")
    df_in = pd.read_csv("sample_tasks.csv")

validate_csv(df_in)

# =========================
# Analysis
# =========================

df, critical_path, graph = analyze_project_from_df(df_in, today)
planned, samples = monte_carlo_project_duration(df, graph, sims=sims)
summary = summarize_samples(planned, samples)

# =========================
# Output
# =========================

c1, c2, c3, c4 = st.columns(4)
c1.metric("Planned (days)", summary["planned_days"])
c2.metric("P50", summary["p50_days"])
c3.metric("P80", summary["p80_days"])
c4.metric("Delay %", summary["delay_probability_pct"])

st.markdown(f"**Critical Path:** {' → '.join(critical_path)}")

st.dataframe(
    df[
        [
            "Task",
            "Start",
            "Due",
            "Progress",
            "Base Risk %",
            "Propagated Risk %",
            "On Critical Path",
        ]
    ],
    use_container_width=True,
)

left, right = st.columns(2)
with left:
    st.pyplot(make_task_risk_fig(df), clear_figure=True)
with right:
    st.pyplot(make_monte_carlo_fig(samples, planned), clear_figure=True)
