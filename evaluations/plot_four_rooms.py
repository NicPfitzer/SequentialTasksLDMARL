#!/usr/bin/env python3
"""
Standalone plotting script for four_rooms experiments.
Reads summary_seeds.csv and generates plots without rerunning experiments.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- defaults ---------------------------
OUT_ROOT = Path(__file__).resolve().parent / "experiments" / "four_rooms_summary_repeated"
CSV_DEFAULT = OUT_ROOT / "summary_seeds.csv"

LABEL_MAP = {
    "naive_rl": "Vanilla RL",
    "rl": "Tuned Rewards RL",
    "rollout": "Ours",
}

PLOT_FILENAMES = {
    "on_goal": "on_goal_mean_ci",
    "opened_final_room": "opened_final_room_mean_ci",
    "success_level": "success_level_mean_ci",
}

MODEL_COLORS = {
    "naive_rl": "#1f77b4",   # blue
    "rl": "#ff7f0e",         # orange
    "rollout": "#2ca02c",    # green
}

# --------------------------- fonts ---------------------------
# plt.rcParams.update({
#     "pdf.fonttype": 42,
#     "ps.fonttype": 42,
#     "font.family": "serif",    # use serif fonts
#     "mathtext.fontset": "cm",  # Computer Modern for math
#     "font.serif": ["cmr10"],   # Computer Modern Roman
# })
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
})
# --------------------------- helpers ---------------------------
def pooled_mean_std(group: pd.DataFrame, mean_col: str, std_col: str, n_col: str) -> pd.Series:
    """Pooled mean/std across seeds given per-seed mean/std and n."""
    means = group[mean_col].to_numpy(dtype=float)
    stds = group[std_col].to_numpy(dtype=float)
    ns = group[n_col].to_numpy(dtype=float)
    N = ns.sum()
    if N <= 1 or len(means) == 0:
        M = float(np.mean(means)) if len(means) else 0.0
        return pd.Series({"pooled_mean": M, "pooled_std": 0.0})
    M = float((means * ns).sum() / N)                   # pooled mean
    ssw = float(((ns - 1.0) * (stds ** 2)).sum())       # within-seed SS
    ssb = float((ns * ((means - M) ** 2)).sum())        # between-seed SS
    var = (ssw + ssb) / max(N - 1.0, 1.0)
    return pd.Series({"pooled_mean": M, "pooled_std": float(np.sqrt(var))})

def plot_metric(df: pd.DataFrame, metric_key: str, outdir: Path) -> None:
    mean_col = f"{metric_key}_mean"
    std_col = f"{metric_key}_std"
    if mean_col not in df.columns or std_col not in df.columns:
        print(f"[skip] Missing columns for {metric_key}: {mean_col}, {std_col}")
        return

    dfx = df.dropna(subset=[mean_col, std_col]).copy()
    if dfx.empty:
        print(f"[skip] No data for {metric_key}.")
        return

    dfx["checkpoint_step"] = pd.to_numeric(dfx["checkpoint_step"], errors="coerce")
    dfx = dfx.dropna(subset=["checkpoint_step"]).sort_values(["model", "checkpoint_step", "seed"])

    pooled = (
        dfx.groupby(["model", "checkpoint_step"], as_index=False)
           .apply(lambda g: pooled_mean_std(g, mean_col, std_col, "episodes"))
           .reset_index(drop=True)
           .sort_values(["model", "checkpoint_step"])
    )

    fig = plt.figure(figsize=(9, 7.75))
    ax = plt.gca()
    for model, g in pooled.groupby("model"):
        x = g["checkpoint_step"].to_numpy()
        y = g["pooled_mean"].to_numpy()
        s = g["pooled_std"].to_numpy()
        ax.plot(x, y, "-o", label=LABEL_MAP.get(model, model), color=MODEL_COLORS[model], linewidth=2.0, markersize=6)
        ax.fill_between(x, y - s, y + s, color=MODEL_COLORS[model],alpha=0.15)

    ax.set_xlabel("Frames", fontsize=18)
    ylabel = {
        "on_goal": "On-goal (mean ± std)",
        "opened_final_room": "Found final room (mean ± std)",
        "success_level": "Rooms found (mean ± std)",
    }[metric_key]
    ax.set_ylabel(ylabel, fontsize=18)
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Model", loc="best", frameon=True, fontsize=18, title_fontsize=18)
    fig.tight_layout()

    base = outdir / PLOT_FILENAMES[metric_key]
    fig.savefig(base.with_suffix(".png"), dpi=150)
    fig.savefig(base.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[ok] Saved → {base.with_suffix('.png')} and {base.with_suffix('.pdf')}")

# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser(description="Plot pooled metrics from summary CSV.")
    ap.add_argument("--csv", type=Path, default=CSV_DEFAULT, help="Path to summary_seeds.csv")
    ap.add_argument("--outdir", type=Path, default=OUT_ROOT, help="Output directory for plots")
    args = ap.parse_args()

    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    required = {"model", "checkpoint_step", "seed", "episodes"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    plot_metric(df, "on_goal", outdir)
    plot_metric(df, "opened_final_room", outdir)
    plot_metric(df, "success_level", outdir)

if __name__ == "__main__":
    main()
