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
        try:
            os.remove(path)
        except Exception:
            pass


def validate_csv(df):
    required = ["Task", "Start", "Due", "Progress"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {', '.join(missing)}")
        st.stop()

    # Optional PERT validation (only if any PERT column exists)
    pert_cols = ["Optimistic", "MostLikely", "Pessimistic"]
    has_any_pert = any(c in df.columns for c in pert_cols)

    if has_any_pert:
        for c in pert_cols:
            if c not in df.columns:
                st.error(f"If you use PERT, your CSV must include all three columns: {', '.join(pert_cols)}")
                st.stop()

        bad_rows = []
        for i, r in df.iterrows():
            vals = [r.get("Optimistic"), r.get("MostLikely"), r.get("Pessimistic")]
            # If row doesn't use PERT, skip
            if all(pd.isna(v) or str(v).strip() == "" for v in vals):
                continue
            try:
                o = float(r["Optimistic"])
                m = float(r["MostLikely"])
                p = float(r["Pessimistic"])
                if not (o <= m <= p):
                    bad_rows.append((i + 2, r["Task"], "Must satisfy Optimistic <= MostLikely <= Pessimistic"))
            except Exception:
                bad_rows.append((i + 2, r["Task"], "PERT values must be numeric"))

        if bad_rows:
            msg = "Invalid PERT rows found (CSV row numbers shown):\n\n"
            msg += "\n".join([f"- Row {rn} ({task}): {reason}" for rn, task, reason in bad_rows[:15]])
            st.error(msg)
            st.stop()


# =========================
# UI
# =========================

st.set_page_config(page_title="APRA Dashboard", layout="wide")
st.title("APRA — Project Risk Dashboard")

with st.sidebar:
    uploaded = st.file_uploader("Upload tasks CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data", value=False)
    sims = st.slider("Monte Carlo simulations", 500, 20000, 1000, step=500)  # safer default for cloud
    today = st.date_input("Today", value=date.today())

    # Sample download
    try:
        with open("sample_tasks.csv", "rb") as f:
            st.download_button(
                "Download sample CSV",
                f,
                file_name="sample_tasks.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.caption("sample_tasks.csv not found in repo (optional).")

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

# PERT detection banner
if all(c in df_in.columns for c in ["Optimistic", "MostLikely", "Pessimistic"]):
    st.success("PERT enabled: Monte Carlo will use Optimistic/MostLikely/Pessimistic where provided.")
else:
    st.info("PERT not detected. Monte Carlo uses Start/Due durations + risk model.")

# =========================
# Analysis
# =========================

df, critical_path, graph = analyze_project_from_df(df_in, today)

with st.spinner("Running Monte Carlo..."):
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

# Dynamic table (show PERT cols if present)
pert_cols = ["Optimistic", "MostLikely", "Pessimistic"]
if all(c in df_in.columns for c in pert_cols):
    display_cols = ["Task", "Start", "Due", "Progress"] + pert_cols + ["Base Risk %", "Propagated Risk %", "On Critical Path"]
else:
    display_cols = ["Task", "Start", "Due", "Progress", "Base Risk %", "Propagated Risk %", "On Critical Path"]

st.dataframe(df[display_cols], use_container_width=True)

left, right = st.columns(2)
with left:
    st.pyplot(make_task_risk_fig(df), clear_figure=True)
with right:
    st.pyplot(make_monte_carlo_fig(samples, planned), clear_figure=True)
