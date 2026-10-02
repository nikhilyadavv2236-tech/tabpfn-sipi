"""Create publication-ready figures and statistical summaries from experiment runs.

Uses identical training draws (matched by N and seed) to compare TabPFN and GP
with a two-sided paired t-test. Learning-curve error bars are mean +/- 1 SD
over random training draws, not uncertainty across the fixed test rows.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ttest_rel

COLORS = {"TabPFN": "#d62728", "XGBoost": "#1f77b4", "MLP": "#2ca02c", "GP": "#9467bd"}
MARKERS = {"TabPFN": "o", "XGBoost": "s", "MLP": "^", "GP": "D"}


def _format_p_value(p_value):
    return "<0.001" if p_value < 0.001 else f"{p_value:.3f}"


def paired_tabpfn_vs_gp(df):
    """Return paired accuracy comparison; lower RMSE means TabPFN wins."""
    rows = []
    for n, sub in df.groupby("N"):
        paired = sub.pivot(index="seed", columns="model", values="rmse")
        if not {"TabPFN", "GP"}.issubset(paired.columns):
            continue
        difference = paired["TabPFN"] - paired["GP"]
        test = ttest_rel(paired["TabPFN"], paired["GP"])
        rows.append({
            "N": n,
            "n_paired_seeds": len(difference),
            "tabpfn_wins": int((difference < 0).sum()),
            "gp_wins": int((difference > 0).sum()),
            "rmse_difference_mean_db": difference.mean(),
            "rmse_difference_std_db": difference.std(),
            "paired_t_statistic": test.statistic,
            "paired_t_p_value_two_sided": test.pvalue,
            "cohen_dz": difference.mean() / difference.std() if difference.std() else float("nan"),
        })
    return pd.DataFrame(rows)


def main():
    Path("results").mkdir(exist_ok=True)
    Path("figures").mkdir(exist_ok=True)
    pdf_dir = Path("output/pdf")
    pdf_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv("results/raw_results.csv")

    # Mean +/- SD across the matched random few-shot training subsets.
    agg = df.groupby(["model", "N"]).agg(
        rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
        mae_mean=("mae", "mean"), mae_std=("mae", "std"),
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        fit_time_mean_s=("fit_time_s", "mean"),
        predict_time_mean_s=("predict_time_s", "mean"),
    ).reset_index()
    agg["total_time_mean_s"] = agg["fit_time_mean_s"] + agg["predict_time_mean_s"]
    agg.to_csv("results/metrics_by_budget.csv", index=False)

    # Figure: mean +/- 1 SD across the ten matched training draws.
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for model in agg["model"].unique():
        sub = agg[agg["model"] == model].sort_values("N")
        ax.errorbar(
            sub["N"], sub["rmse_mean"], yerr=sub["rmse_std"],
            label=model, marker=MARKERS.get(model, "o"), color=COLORS.get(model),
            capsize=3, linewidth=1.5,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Training budget N (log scale)")
    ax.set_ylabel("Held-out RMSE (dB)")
    ax.set_title("Few-shot insertion-loss regression (mean +/- 1 SD, 10 seeds)")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("figures/rmse_vs_n.png", dpi=240)
    fig.savefig(pdf_dir / "rmse_vs_n.pdf", bbox_inches="tight")
    plt.close(fig)

    stats = paired_tabpfn_vs_gp(df)
    if not stats.empty:
        stats.to_csv("results/tabpfn_vs_gp_paired_tests.csv", index=False)

    # Compact tables for the paper plus a complete machine-readable table.
    rmse_table = agg.pivot(index="N", columns="model", values="rmse_mean").round(4)
    mae_table = agg.pivot(index="N", columns="model", values="mae_mean").round(4)
    r2_table = agg.pivot(index="N", columns="model", values="r2_mean").round(4)
    total_time_table = agg.pivot(index="N", columns="model", values="total_time_mean_s").round(4)
    rmse_table.to_csv("results/summary_table.csv")

    uncertainty_table = None
    if {"picp_90", "mpiw_90_db"}.issubset(df.columns):
        uncertainty = df.dropna(subset=["picp_90"]).groupby(["N", "model"]).agg(
            picp_90_mean=("picp_90", "mean"),
            picp_90_std=("picp_90", "std"),
            mpiw_90_db_mean=("mpiw_90_db", "mean"),
        ).reset_index()
        if not uncertainty.empty:
            uncertainty.to_csv("results/uncertainty_calibration.csv", index=False)
            uncertainty_table = uncertainty.pivot(index="N", columns="model", values="picp_90_mean").round(4)

            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            for model in uncertainty["model"].unique():
                sub = uncertainty[uncertainty["model"] == model].sort_values("N")
                ax.errorbar(
                    sub["N"], sub["picp_90_mean"], yerr=sub["picp_90_std"],
                    label=model, marker=MARKERS.get(model, "o"), color=COLORS.get(model),
                    capsize=3, linewidth=1.5,
                )
            ax.axhline(0.90, color="black", linestyle="--", linewidth=1, label="Nominal 90%")
            ax.set_xscale("log")
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Training budget N (log scale)")
            ax.set_ylabel("90% prediction-interval coverage")
            ax.set_title("Uncertainty calibration (mean +/- 1 SD)")
            ax.legend()
            ax.grid(alpha=0.3, which="both")
            fig.tight_layout()
            fig.savefig("figures/picp90_vs_n.png", dpi=240)
            fig.savefig(pdf_dir / "picp90_vs_n.pdf", bbox_inches="tight")
            plt.close(fig)

    with open("results/summary_table.md", "w", encoding="utf-8") as f:
        f.write("# Test RMSE (dB), mean across seeds\n\n")
        f.write(rmse_table.to_markdown())
        f.write("\n\n# Test MAE (dB), mean across seeds\n\n")
        f.write(mae_table.to_markdown())
        f.write("\n\n# Test R², mean across seeds\n\n")
        f.write(r2_table.to_markdown())
        f.write("\n\n# Total wall-clock per run (fit + prediction, seconds)\n\n")
        f.write(total_time_table.to_markdown())
        if uncertainty_table is not None:
            f.write("\n\n# 90% prediction-interval coverage (PICP; nominal = 0.90)\n\n")
            f.write(uncertainty_table.to_markdown())
        if not stats.empty:
            report = stats[["N", "tabpfn_wins", "gp_wins", "rmse_difference_mean_db", "paired_t_p_value_two_sided", "cohen_dz"]].copy()
            report["paired_t_p_value_two_sided"] = report["paired_t_p_value_two_sided"].map(_format_p_value)
            f.write("\n\n# TabPFN vs GP: paired RMSE comparison\n\n")
            f.write(report.to_markdown(index=False))
            f.write("\n\nTwo-sided paired t-tests use the same seed-specific training draws; negative differences favor TabPFN.\n")

    print("Saved figures/rmse_vs_n.png, output/pdf/rmse_vs_n.pdf")
    print("Saved results/metrics_by_budget.csv, results/summary_table.md")
    if not stats.empty:
        print("Saved results/tabpfn_vs_gp_paired_tests.csv")
        print(stats[["N", "tabpfn_wins", "gp_wins", "paired_t_p_value_two_sided"]].to_string(index=False))
    if uncertainty_table is not None:
        print("Saved results/uncertainty_calibration.csv, figures/picp90_vs_n.png, output/pdf/picp90_vs_n.pdf")


if __name__ == "__main__":
    main()
