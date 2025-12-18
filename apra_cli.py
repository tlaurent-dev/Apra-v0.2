import argparse
import pandas as pd
from apra_core import analyze_project, monte_carlo_project_duration, summarize_samples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to tasks CSV")
    parser.add_argument("--sims", type=int, default=5000, help="Monte Carlo simulations")
    parser.add_argument("--base", action="store_true", help="Use base risk (default uses propagated risk)")
    args = parser.parse_args()

    df, critical_path, graph = analyze_project(args.csv)
    df.to_csv("results.csv", index=False)

    planned, samples = monte_carlo_project_duration(
        df, graph, sims=args.sims, use_propagated=(not args.base), seed=42
    )
    summary = summarize_samples(planned, samples)

    pd.DataFrame({"project_duration_days": samples}).to_csv("mc_samples.csv", index=False)

    print("\n=== APRA POC RESULTS ===")
    print(df[["Task", "Base Risk %", "Propagated Risk %", "On Critical Path"]])
    print("\nCritical Path:", " -> ".join(critical_path))

    print("\n=== MONTE CARLO FORECAST ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSaved: results.csv")
    print("Saved: mc_samples.csv")

if __name__ == "__main__":
    main()
