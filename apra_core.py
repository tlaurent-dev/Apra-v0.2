import pandas as pd
import random
from datetime import date, datetime

# ======================================================
# Base risk calculation
# ======================================================

def calculate_base_risk(row, today):
    start = pd.to_datetime(row["Start"]).date()
    due = pd.to_datetime(row["Due"]).date()
    progress = float(row["Progress"])

    if progress >= 100:
        return 0.0

    if today >= due:
        return 100.0

    total_days = max((due - start).days, 1)
    elapsed_days = max((today - start).days, 0)

    expected_progress = (elapsed_days / total_days) * 100
    risk = max(expected_progress - progress, 0)

    return round(min(risk, 100), 2)


# ======================================================
# Dependency handling
# ======================================================

def build_dependency_graph(df):
    graph = {}
    for _, row in df.iterrows():
        task = row["Task"]
        deps = []
        if isinstance(row.get("Dependencies"), str) and row["Dependencies"].strip():
            deps = [d.strip() for d in row["Dependencies"].split(",")]
        graph[task] = deps
    return graph


def propagate_risk(df, graph):
    base = dict(zip(df["Task"], df["Base Risk %"] / 100))
    propagated = base.copy()

    changed = True
    while changed:
        changed = False
        for task, deps in graph.items():
            if not deps:
                continue
            no_delay = 1.0
            for d in deps:
                no_delay *= (1 - propagated.get(d, 0))
            new_risk = 1 - no_delay
            if new_risk > propagated[task]:
                propagated[task] = new_risk
                changed = True

    return {k: round(v * 100, 2) for k, v in propagated.items()}


def find_critical_path(df, graph):
    durations = {}
    for _, r in df.iterrows():
        start = pd.to_datetime(r["Start"])
        due = pd.to_datetime(r["Due"])
        durations[r["Task"]] = max((due - start).days, 1)

    longest = {t: durations[t] for t in graph}
    predecessor = {t: None for t in graph}

    changed = True
    while changed:
        changed = False
        for t, deps in graph.items():
            for d in deps:
                if longest[d] + durations[t] > longest[t]:
                    longest[t] = longest[d] + durations[t]
                    predecessor[t] = d
                    changed = True

    end = max(longest, key=longest.get)
    path = []
    while end:
        path.append(end)
        end = predecessor[end]

    return list(reversed(path))


# ======================================================
# Core analysis
# ======================================================

def analyze_project(csv_path, today=None):
    today = today or date.today()
    df = pd.read_csv(csv_path)

    df["Base Risk %"] = df.apply(lambda r: calculate_base_risk(r, today), axis=1)

    graph = build_dependency_graph(df)
    propagated = propagate_risk(df, graph)

    df["Propagated Risk %"] = df["Task"].map(propagated)

    critical_path = find_critical_path(df, graph)
    df["On Critical Path"] = df["Task"].isin(critical_path)

    return df, critical_path, graph


# ======================================================
# Monte Carlo helpers
# ======================================================

def topo_order(graph):
    indeg = {t: 0 for t in graph}
    for t, deps in graph.items():
        for d in deps:
            indeg[t] += 1

    queue = [t for t in indeg if indeg[t] == 0]
    order = []

    while queue:
        n = queue.pop()
        order.append(n)
        for child, deps in graph.items():
            if n in deps:
                indeg[child] -= 1
                if indeg[child] == 0:
                    queue.append(child)

    return order if len(order) == len(graph) else list(graph.keys())


def sample_task_duration(base_days, risk_pct):
    """
    Original risk-driven sampling (fallback when no PERT).
    """
    r = max(0.0, min(risk_pct / 100.0, 1.0))
    roll = random.random()

    if roll < 1 - r:
        factor = 1.0
    elif roll < 1 - r * 0.55:
        factor = 1.15
    elif roll < 1 - r * 0.20:
        factor = 1.35
    else:
        factor = 1.75

    return base_days * factor


def _to_float_or_none(x):
    try:
        if pd.isna(x):
            return None
        s = str(x).strip()
        if s == "":
            return None
        return float(s)
    except Exception:
        return None


def pert_sample(o, m, p):
    """
    PERT sampling via Beta distribution, bounded to [o, p].
    Assumes o <= m <= p.
    """
    import random

    if o is None or m is None or p is None:
        return None
    if not (o <= m <= p):
        return None
    if p == o:
        return float(m)

    alpha = 1.0 + 4.0 * (m - o) / (p - o)
    beta = 1.0 + 4.0 * (p - m) / (p - o)

    u = random.betavariate(alpha, beta)
    return o + u * (p - o)


def task_duration_for_mc(row, base_duration_days, risk_pct):
    """
    If Optimistic/MostLikely/Pessimistic are present and valid -> sample PERT duration.
    Else fallback to original risk-driven sampling.
    """
    o = _to_float_or_none(row.get("Optimistic"))
    m = _to_float_or_none(row.get("MostLikely"))
    p = _to_float_or_none(row.get("Pessimistic"))

    sampled = pert_sample(o, m, p)
    if sampled is not None:
        # modest stretch based on risk so risk signal still matters
        r = max(0.0, min(risk_pct / 100.0, 1.0))
        stretch = 1.0 + 0.30 * r
        return max(0.5, sampled * stretch)

    return sample_task_duration(base_duration_days, risk_pct)


def monte_carlo_project_duration(df, graph, sims=5000, use_propagated=True, seed=42):
    random.seed(seed)

    # Base durations from Start/Due (days)
    durations = {}
    for _, r in df.iterrows():
        start = pd.to_datetime(r["Start"])
        due = pd.to_datetime(r["Due"])
        durations[r["Task"]] = max((due - start).days, 1)

    risk_col = "Propagated Risk %" if use_propagated else "Base Risk %"
    risk_map = dict(zip(df["Task"], df[risk_col]))

    order = topo_order(graph)
    samples = []

    # Planned duration for reporting:
    # - If PERT exists & valid, use mean (o+4m+p)/6
    # - Else use Start/Due duration
    planned = 0.0
    for _, r in df.iterrows():
        o = _to_float_or_none(r.get("Optimistic"))
        m = _to_float_or_none(r.get("MostLikely"))
        p = _to_float_or_none(r.get("Pessimistic"))
        if o is not None and m is not None and p is not None and (o <= m <= p):
            planned += (o + 4 * m + p) / 6.0
        else:
            planned += durations[r["Task"]]

    for _ in range(sims):
        finish = {}
        for t in order:
            deps = graph.get(t, [])
            start_time = max([finish.get(d, 0.0) for d in deps], default=0.0)

            row = df.loc[df["Task"] == t].iloc[0]
            dur = task_duration_for_mc(row, durations[t], risk_map.get(t, 0.0))

            finish[t] = start_time + dur

        samples.append(max(finish.values()) if finish else 0.0)

    return planned, samples


def summarize_samples(planned, samples):
    s = sorted(samples)
    n = len(s)

    def pct(p):
        return s[int(p * (n - 1))]

    return {
        "planned_days": round(planned, 2),
        "p50_days": round(pct(0.5), 2),
        "p80_days": round(pct(0.8), 2),
        "p90_days": round(pct(0.9), 2),
        "delay_probability_pct": round(sum(x > planned for x in s) / n * 100, 2),
    }


# ======================================================
# Streamlit-safe figures (no disk writes)
# ======================================================

def make_task_risk_fig(df):
    import matplotlib.pyplot as plt
    d = df.sort_values("Propagated Risk %", ascending=False)
    fig = plt.figure(figsize=(7.5, 3.5))
    plt.bar(d["Task"], d["Propagated Risk %"])
    plt.title("Task Risk (Propagated)")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    return fig


def make_monte_carlo_fig(samples, planned):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.5, 3.5))
    plt.hist(samples, bins=35)
    plt.axvline(planned, linestyle="--", linewidth=2)
    plt.title("Monte Carlo Project Duration")
    plt.tight_layout()
    return fig
