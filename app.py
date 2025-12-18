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


def ensure_pert_columns(df):
    # Ensure these exist so we can override even if CSV didn't include them
    for c in ["Optimistic", "MostLikely", "Pessimistic"]:
        if c not in df.columns:
            df[c] = ""
    return df


def init_overrides():
    if "overrides" not in st.session_state:
        # overrides: {task_name: {"Optimistic": x, "MostLikely": y, "Pessimistic": z, "Progress": p}}
        st.session_state["overrides"] = {}


def apply_overrides(df):
    df2 = df.copy()
    overrides = st.session_state.get("overrides", {})
    if not overrides:
        return df2

    for i, r in df2.iterrows():
        t = r["Task"]
        if t in overrides:
            ov = overrides[t]
            for k, v in ov.items():
                df2.at[i, k] = v
    return df2


# =========================
# UI
# =========================

st.set_page_config(page_title="APRA Dashboard", layout="wide")
st.title("APRA — Project Risk Dashboard")

init_overrides()

with st.sidebar:
    st.header("Inputs")
    uploaded = st.file_uploader("Upload tasks CSV", type=["csv"])
    use_demo = st.checkbox("Use demo data", value=False)
    sims = st.slider("Monte Carlo simulations", 500, 20000, 1000, step=500)  # cloud-safe default
    today = st.date_input("Today", value=date.today())

    st.divider()
    st.header("Samples")
    try:
        with open("sample_tasks.csv", "rb") as f:
            st.download_button("Download sample CSV", f, file_name="sample_tasks.csv", mime="text/csv")
    except FileNotFoundError:
        st.caption("sample_tasks.csv not found in repo (optional).")

    try:
        with open("sample_tasks_pert.csv", "rb") as f:
            st.download_button("Download PERT sample CSV", f, file_name="sample_tasks_pert.csv", mime="text/csv")
    except FileNotFoundError:
        st.caption("sample_tasks_pert.csv not found in repo (recommended).")

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
df_in = ensure_pert_columns(df_in)

# =========================
# What-if controls (PERT overrides)
# =========================

st.subheader("What-if controls (PERT overrides)")
st.caption("Override Optimistic / MostLikely / Pessimistic (and optionally Progress) for a single task. This does not edit your CSV file.")

colA, colB = st.columns([2, 1])

task_list = df_in["Task"].astype(str).tolist()
selected_task = colA.selectbox("Select task to override", task_list, index=0 if task_list else None)

with colB:
    if st.button("Reset ALL overrides"):
        st.session_state["overrides"] = {}
        st.success("Overrides cleared.")

if selected_task:
    row = df_in.loc[df_in["Task"].astype(str) == str(selected_task)].iloc[0]

    # Current values (may be empty if CSV didn't include PERT)
    def _val(x, default):
        try:
            if pd.isna(x) or str(x).strip() == "":
                return default
            return float(x)
        except Exception:
            return default

    cur_o = _val(row.get("Optimistic", ""), 1.0)
    cur_m = _val(row.get("MostLikely", ""), max(cur_o, 2.0))
    cur_p = _val(row.get("Pessimistic", ""), max(cur_m, 3.0))
    cur_prog = float(row.get("Progress", 0))

    st.markdown(f"**Selected:** `{selected_task}`")

    c1, c2, c3, c4 = st.columns(4)
    new_o = c1.number_input("Optimistic (days)", min_value=0.5, value=float(cur_o), step=0.5)
    new_m = c2.number_input("Most Likely (days)", min_value=0.5, value=float(cur_m), step=0.5)
    new_p = c3.number_input("Pessimistic (days)", min_value=0.5, value=float(cur_p), step=0.5)
    new_prog = c4.slider("Progress (%)", 0, 100, int(round(cur_prog)), step=1)

    if not (new_o <= new_m <= new_p):
        st.error("Invalid PERT override: must satisfy Optimistic ≤ MostLikely ≤ Pessimistic.")
    else:
        colS1, colS2 = st.columns([1, 2])
        if colS1.button("Apply override for this task"):
            st.session_state["overrides"][str(selected_task)] = {
                "Optimistic": float(new_o),
                "MostLikely": float(new_m),
                "Pessimistic": float(new_p),
                "Progress": float(new_prog),
            }
            st.success("Override applied.")

        if colS2.button("Remove override for this task"):
            st.session_state["overrides"].pop(str(selected_task), None)
            st.success("Override removed.")

# Apply overrides to an analysis copy
df_for_analysis = apply_overrides(df_in)

# =========================
# PERT detection banner (after overrides)
# =========================

if all(c in df_for_analysis.columns for c in ["Optimistic", "MostLikely", "Pessimistic"]):
    # detect if at least one row actually has all three values filled
    has_any_row_pert = False
    for _, r in df_for_analysis.iterrows():
        vals = [r.get("Optimistic"), r.get("MostLikely"), r.get("Pessimistic")]
        if all(not (pd.isna(v) or str(v).strip() == "") for v in vals):
            has_any_row_pert = True
            break
    if has_any_row_pert:
        st.success("PERT enabled: Monte Carlo will use Optimistic/MostLikely/Pessimistic where provided (including overrides).")
    else:
        st.info("PERT columns exist, but no rows have PERT values filled. Using Start/Due durations + risk model.")
else:
    st.info("PERT not detected. Monte Carlo uses Start/Due durations + risk model.")

# =========================
# Analysis
# =========================

df, critical_path, graph = analyze_project_from_df(df_for_analysis, today)

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

st.subheader("Task Table")
display_cols = ["Task", "Start", "Due", "Progress", "Optimistic", "MostLikely", "Pessimistic", "Base Risk %", "Propagated Risk %", "On Critical Path"]
st.dataframe(df[display_cols], use_container_width=True)

left, right = st.columns(2)
with left:
    st.pyplot(make_task_risk_fig(df), clear_figure=True)
with right:
    st.pyplot(make_monte_carlo_fig(samples, planned), clear_figure=True)

# Show current overrides for transparency
st.subheader("Active overrides")
if st.session_state["overrides"]:
    st.json(st.session_state["overrides"])
else:
    st.caption("No overrides applied.")
