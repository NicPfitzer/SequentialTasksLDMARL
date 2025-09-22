#!/usr/bin/env python3
"""
Standalone plotting script for four_rooms_initialization.
Reads summary_seeds.csv and generates grouped error-bar plots by config_type.

Usage (defaults are fine):
  python plot_initialization_from_csv.py
  # or override:
  python plot_initialization_from_csv.py --csv /path/to/summary_seeds.csv --outdir /path/to/out
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# --------------------------- defaults ---------------------------
OUT_ROOT = Path(__file__).resolve().parent / "experiments" / "four_rooms_initialization"
CSV_DEFAULT = OUT_ROOT / "summary_seeds.csv"

# Consistent pretty labels (matches your aggregation script intent)
LABEL_MAP = {
    "vanilla_rl": "Vanilla RL",
    "naive_rl": "Vanilla RL",
    "rl": "Tuned Rewards RL",
    "rollout_2": "Ours - No Language Init.",
    "rollout_3": "Ours",
    # passthrough for others
}

# Fixed colors per model label (so colors don't depend on order)
MODEL_COLORS = {
    "Vanilla RL": "#1f77b4",                 # blue
    "Tuned Rewards RL": "#ff7f0e",           # orange
    "Ours": "#2ca02c",       # green
    "Ours - No Language Init.": "#d62728",    # red
}

# Desired config display order
CONFIG_ORDER = ["leftmost", "room 1", "room 2", "rightmost", "random", "even"]

# Which metrics to plot (grouped by config)
METRICS = {
    "success_level": "Rooms Found (mean ± std)",
    "on_goal": "On-goal (mean ± std)",
    "opened_final_room": "Found final room (mean ± std)",
}

# --------------------------- fonts & ticks ---------------------------
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 18,            # base font
    "axes.labelsize": 18,       # x/y labels
    "axes.titlesize": 18,       # titles
    "legend.fontsize": 18,      # legend text
    "legend.title_fontsize": 18,# legend title
    "xtick.labelsize": 18,      # axis values
    "ytick.labelsize": 18,
})


_plain_formatter = ScalarFormatter(useOffset=False, useMathText=False)
_plain_formatter.set_scientific(False)

# --------------------------- helpers ---------------------------
def pooled_mean_std(group: pd.DataFrame, mean_col: str, std_col: str, n_col: str) -> pd.Series:
    """Pooled mean/std across seeds given per-seed mean/std and n episodes."""
    means = group[mean_col].to_numpy(dtype=float)
    stds  = group[std_col].to_numpy(dtype=float)
    ns    = group[n_col].to_numpy(dtype=float)
    N = ns.sum()
    if N <= 1 or means.size == 0:
        M = float(np.mean(means)) if means.size else 0.0
        return pd.Series({"pooled_mean": M, "pooled_std": 0.0, "N": int(N)})
    M = float((means * ns).sum() / N)
    ssw = float(((ns - 1.0) * (stds ** 2)).sum())      # within
    ssb = float((ns * ((means - M) ** 2)).sum())       # between
    var = (ssw + ssb) / max(N - 1.0, 1.0)
    return pd.Series({"pooled_mean": M, "pooled_std": float(np.sqrt(var)), "N": int(N)})

def pretty_model_name(raw: str) -> str:
    return LABEL_MAP.get(raw, raw)

def compute_pooled(df: pd.DataFrame, metric_key: str) -> pd.DataFrame:
    mean_col = f"{metric_key}_mean"
    std_col  = f"{metric_key}_std"
    for col in (mean_col, std_col, "episodes", "model", "config_type"):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column for {metric_key}: {col}")

    tmp = df.dropna(subset=[mean_col, std_col]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["config_type", "model", "pooled_mean", "pooled_std", "N"])

    tmp["model"] = tmp["model"].map(pretty_model_name)

    pooled = (
        tmp.groupby(["config_type", "model"], as_index=False)
           .apply(lambda g: pooled_mean_std(g, mean_col, std_col, "episodes"))
           .reset_index(drop=True)
    )
    return pooled

def plot_grouped(df_pooled: pd.DataFrame, metric_key: str, outdir: Path) -> None:
    if df_pooled.empty:
        print(f"[skip] No data for {metric_key}.")
        return

    # Order configs and models
    present_configs = list(df_pooled["config_type"].unique())
    config_order = [c for c in CONFIG_ORDER if c in present_configs] or sorted(present_configs)

    desired_models = [
        "Vanilla RL",
        "Tuned Rewards RL",
        "Ours",
    ]
    present_models = [m for m in desired_models if m in set(df_pooled["model"])]
    if not present_models:
        present_models = sorted(df_pooled["model"].unique())

    # Prepare grouped positions
    x = np.arange(len(config_order), dtype=float)
    n_models = len(present_models)
    group_width = 0.35
    step = group_width / max(n_models, 1)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * step

    # Figure
    width = 1.5 * len(config_order) + 2.5
    height = width / 1.16
    fig = plt.figure(figsize=(9, 7.75))
    ax = plt.gca()

    cap = 4
    for i, model in enumerate(present_models):
        sub = df_pooled[df_pooled["model"] == model].set_index("config_type")
        means = [sub.loc[c, "pooled_mean"] if c in sub.index else np.nan for c in config_order]
        stds  = [sub.loc[c, "pooled_std"]  if c in sub.index else np.nan for c in config_order]
        xi = x + offsets[i]
        color = MODEL_COLORS.get(model, None)

        ax.errorbar(
            xi, means, yerr=stds,
            fmt="o", capsize=cap, elinewidth=1.4, linewidth=0.0,
            markersize=7, label=model, color=color,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(config_order, rotation=0, fontsize=18)
    ax.set_xlabel("Initialization config", fontsize=18)
    ax.set_ylabel(METRICS[metric_key], fontsize=18)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(title="Model", loc="lower center", frameon=True, fontsize=18, title_fontsize=18)

    # Ticks and number formatting
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.yaxis.set_major_formatter(_plain_formatter)

    fig.tight_layout()
    base = outdir / f"{metric_key}_grouped_by_config"
    fig.savefig(base.with_suffix(".png"), dpi=150)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] Saved → {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")

# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot grouped metrics by config from summary CSV.")
    ap.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="Path to summary_seeds.csv")
    ap.add_argument("--outdir", type=Path, default=OUT_ROOT, help="Output directory for plots")
    ap.add_argument("--metrics", nargs="*", default=list(METRICS.keys()),
                    help=f"Subset of metrics to plot. Choices: {list(METRICS.keys())}")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Basic required columns
    required = {"model", "config_type", "episodes"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # For each requested metric, compute pooled stats and plot
    for metric_key in args.metrics:
        mean_col = f"{metric_key}_mean"
        std_col  = f"{metric_key}_std"
        if mean_col not in df.columns or std_col not in df.columns:
            print(f"[skip] Missing {mean_col}/{std_col}; skipping {metric_key}.")
            continue
        pooled = compute_pooled(df, metric_key)
        plot_grouped(pooled, metric_key, outdir)

if __name__ == "__main__":
    main()
