import os
import re
from math import sqrt
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment
from benchmarl.experiment import Experiment

# --------------------------- config ---------------------------------
NUM_ROLLOUTS = 500
DEVICE = "cpu"

# Repeat control
N_EXPERIMENTS = 1              # <- how many repeats
BASE_SEED = 42                  # <- starting seed
SEEDS = [BASE_SEED + i for i in range(N_EXPERIMENTS)]  # deterministic sequence

# Paths
RESTORE_PATH_ROOT = (
    "/Users/nicolaspfitzer/ProrokLab/SequentialTasks/checkpoints/"
    "four_rooms/"
)
MODEL_FOLDERS = [
    "four_rooms_naive/",
    "four_rooms_rl/",
    "four_rooms_rollout/",
]

# Output
OUT_ROOT = Path(os.path.dirname(os.path.realpath(__file__))) / "experiments" / "four_rooms_summary_repeated"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_ROOT / "summary_seeds.csv"
PLOT_GOAL = OUT_ROOT / "on_goal_mean_ci.png"
PLOT_OFR  = OUT_ROOT / "opened_final_room_mean_ci.png"
PLOT_SL   = OUT_ROOT / "success_level_mean_ci.png"
# --------------------------------------------------------------------


# --------------------------- helpers ---------------------------------
def pretty_model_label(path_suffix: str) -> str:
    name = path_suffix.strip("/").lower()
    if "naive" in name:
        return "naive_rl"
    if "multitask" in name or "rollout" in name:
        # keep "rollout" label if you prefer
        return "rollout" if "rollout" in name else "multitask"
    if name.endswith("rl"):
        return "rl"
    return name

def parse_frames(fname: str) -> int:
    """Parse frame count from checkpoint_xxxxx.pt"""
    m = re.match(r"checkpoint_(\d+)\.pt", fname)
    return int(m.group(1)) if m else -1

def generate_cfg(
    *,
    config_path: str,
    config_name: str,
    restore_path: str,
    device: str,
    seed: int,
) -> DictConfig:
    """Build Hydra config for a single run, injecting restore path, device, and seed."""
    if not os.path.isabs(restore_path):
        restore_path = os.path.abspath(restore_path)
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=[])
    experiment_name = list(cfg.keys())[0]

    if config_name == "config_four_rooms_rollout":
        # Rollout special-case: policy checkpoint vs base
        cfg[experiment_name].task.params.policy_restore_path = restore_path
        cfg[experiment_name].experiment.restore_file = "checkpoints/four_rooms/four_rooms_rollout/policy_base.pt"
    else:
        cfg[experiment_name].experiment.restore_file = restore_path

    cfg[experiment_name].experiment.restore_map_location = device
    # if your config supports a seed field, set it; otherwise seed is passed to benchmarl below
    try:
        cfg[experiment_name].seed = seed
    except Exception:
        pass
    return cfg[experiment_name]

def run_experiment_once(path_suffix: str, seed: int) -> list[dict]:
    """
    Iterate all checkpoints under a model family folder, evaluate once with a given seed.
    Returns rows with per-checkpoint metrics for this seed, including per-episode mean/std.
    """
    rows = []

    # Special dir mapping (kept from your earlier scripts)
    if path_suffix == "four_rooms_rollout/":
        ckpt_dir = os.path.join(RESTORE_PATH_ROOT, "four_rooms_multitask/")
    else:
        ckpt_dir = os.path.join(RESTORE_PATH_ROOT, path_suffix)

    if not os.path.isdir(ckpt_dir):
        print(f"[warn] Missing directory: {ckpt_dir}")
        return rows

    restore_files = [f for f in os.listdir(ckpt_dir) if (f.endswith(".pt") and f.startswith("checkpoint_"))]
    restore_files = sorted(restore_files, key=parse_frames)

    model_label = pretty_model_label(path_suffix)

    for restore_file in restore_files:
        restore_path = os.path.join(ckpt_dir, restore_file)
        frames = parse_frames(restore_file)
        print(f"[seed={seed}] Evaluating: {model_label} | {restore_file} (frames={frames})")

        cfg = generate_cfg(
            config_path="../conf",
            config_name="config_" + path_suffix[:-1],  # strip trailing slash
            restore_path=restore_path,
            device=DEVICE,
            seed=seed,
        )
        try:
            cfg.experiment.loggers[0] = "csv"
        except Exception:
            pass
        cfg.experiment.render = False
        cfg.experiment.evaluation_episodes = NUM_ROLLOUTS
        cfg.experiment.save_folder = OUT_ROOT / f"{model_label}_frames{frames}_seed{seed}"

        experiment: Experiment = benchmarl_setup_experiment(cfg, seed=seed, main_experiment=False)
        experiment.evaluate()

        agents = experiment.test_env.base_env._env.scenario.world.agents

        # Per-episode arrays
        on_goal = torch.ones((NUM_ROLLOUTS,), device="cpu", dtype=torch.bool)
        opened_final_room = torch.zeros((NUM_ROLLOUTS,), device="cpu", dtype=torch.bool)
        switch_hits = torch.zeros((NUM_ROLLOUTS, 3), device="cpu", dtype=torch.int32)
        for agent in agents:
            on_goal &= agent.on_goal                  # shape [episodes]
            opened_final_room |= agent.switch_hits[:, -1].bool()  # final-step open
            switch_hits |= agent.switch_hits

        on_goal_arr = on_goal.float().numpy()
        ofr_arr = opened_final_room.float().numpy()
        success_level = switch_hits.sum(dim=1).float().numpy()  # how many switches hit

        rows.append(
            {
                "model": model_label,
                "checkpoint_file": restore_file,
                "checkpoint_step": frames,
                # per-episode stats for this seed
                "on_goal_mean": float(on_goal_arr.mean()),
                "on_goal_std": float(on_goal_arr.std(ddof=1)) if NUM_ROLLOUTS > 1 else 0.0,
                "opened_final_room_mean": float(ofr_arr.mean()),
                "opened_final_room_std": float(ofr_arr.std(ddof=1)) if NUM_ROLLOUTS > 1 else 0.0,
                "success_level_mean": float(success_level.mean()),
                "success_level_std": float(success_level.std(ddof=1)) if NUM_ROLLOUTS > 1 else 0.0,
                # metadata
                "episodes": int(cfg.experiment.evaluation_episodes),
                "seed": int(seed),
                "ckpt_path": restore_path,
            }
        )
    return rows

def aggregate_and_plot(df: pd.DataFrame) -> None:
    """Compute pooled mean ± std across all episodes (all seeds) and plot per model vs frames."""
    # ensure numeric frames
    df["checkpoint_step"] = pd.to_numeric(df["checkpoint_step"], errors="coerce")
    df.sort_values(["model", "checkpoint_step", "seed"], inplace=True)

    # save raw (all seeds)
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved per-seed CSV → {SUMMARY_CSV}")
    
    label_map = {
        "naive_rl": "Vanilla RL",
        "rl": "Tuned Rewards RL",
        "rollout": "DeCLaRE",
    }

    def pooled_mean_std(group: pd.DataFrame, mean_col: str, std_col: str, n_col: str):
        """Pooled mean/std across seeds given per-seed mean/std and n."""
        means = group[mean_col].to_numpy()
        stds = group[std_col].to_numpy()
        ns = group[n_col].to_numpy().astype(float)
        N = ns.sum()
        if N <= 1:
            M = means.mean() if len(means) else 0.0
            return pd.Series({"pooled_mean": M, "pooled_std": 0.0})
        # pooled mean
        M = (means * ns).sum() / N
        # within- and between-seed sums of squares
        ssw = ((ns - 1.0) * (stds ** 2)).sum()
        ssb = (ns * ((means - M) ** 2)).sum()
        var = (ssw + ssb) / (N - 1.0)
        return pd.Series({"pooled_mean": M, "pooled_std": float(var ** 0.5)})

    # -------- On-goal pooled mean ± std --------
    g_goal = (
        df.groupby(["model", "checkpoint_step"])
          .apply(lambda g: pooled_mean_std(g, "on_goal_mean", "on_goal_std", "episodes"))
          .reset_index()
          .sort_values(["model", "checkpoint_step"])
    )

    plt.figure(figsize=(9, 5.5))
    for model, gg in g_goal.groupby("model"):
        x = gg["checkpoint_step"].to_numpy()
        y = gg["pooled_mean"].to_numpy()
        s = gg["pooled_std"].to_numpy()
        plt.plot(x, y, "-o", label=label_map.get(model, model))
        plt.fill_between(x, y - s, y + s, alpha=0.15)
    plt.xlabel("Frames")
    plt.ylabel("On-goal (mean ± std)")
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model", loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOT_GOAL, dpi=150)
    plt.savefig(PLOT_GOAL.with_suffix(".pdf"))
    plt.close()
    print(f"Saved plot → {PLOT_GOAL}")

    # -------- Opened final room pooled mean ± std --------
    df_ofr = df.dropna(subset=["opened_final_room_mean", "opened_final_room_std"])
    if not df_ofr.empty:
        g_ofr = (
            df_ofr.groupby(["model", "checkpoint_step"])
                  .apply(lambda g: pooled_mean_std(g, "opened_final_room_mean", "opened_final_room_std", "episodes"))
                  .reset_index()
                  .sort_values(["model", "checkpoint_step"])
        )

        plt.figure(figsize=(9, 5.5))
        for model, gg in g_ofr.groupby("model"):
            x = gg["checkpoint_step"].to_numpy()
            y = gg["pooled_mean"].to_numpy()
            s = gg["pooled_std"].to_numpy()
            plt.plot(x, y, "-o", label=label_map.get(model, model))
            plt.fill_between(x, y - s, y + s, alpha=0.15)
        plt.xlabel("Frames")
        plt.ylabel("Found final room (mean ± std)")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Model", loc="lower right")
        plt.tight_layout()
        plt.savefig(PLOT_OFR, dpi=150)
        plt.savefig(PLOT_OFR.with_suffix(".pdf"))
        plt.close()
        print(f"Saved plot → {PLOT_OFR}")
    else:
        print("No opened_final_room data to plot.")
    
    # ----------- Success level pooled mean ± std -----------
    df_sl = df.dropna(subset=["success_level_mean", "success_level_std"])
    if not df_sl.empty:
        g_sl = (
            df_sl.groupby(["model", "checkpoint_step"])
                  .apply(lambda g: pooled_mean_std(g, "success_level_mean", "success_level_std", "episodes"))
                  .reset_index()
                  .sort_values(["model", "checkpoint_step"])
        )

        plt.figure(figsize=(9, 5.5))
        for model, gg in g_sl.groupby("model"):
            x = gg["checkpoint_step"].to_numpy()
            y = gg["pooled_mean"].to_numpy()
            s = gg["pooled_std"].to_numpy()
            plt.plot(x, y, "-o", label=label_map.get(model, model))
            plt.fill_between(x, y - s, y + s, alpha=0.15)
        plt.xlabel("Frames")
        plt.ylabel("Rooms Found (mean ± std)")
        plt.grid(True, alpha=0.3)
        plt.legend(title="Model", loc="lower right")
        plt.tight_layout()
        plot_sl_path = OUT_ROOT / "success_level_mean_ci.png"
        plt.savefig(plot_sl_path, dpi=150)
        plt.savefig(PLOT_SL.with_suffix(".pdf"))
        plt.close()
        print(f"Saved plot → {plot_sl_path}")
    else:
        print("No success_level data to plot.")


# --------------------------- main ---------------------------------
def main():
    print(f"Seeds: {SEEDS}")
    all_rows = []
    for seed in SEEDS:
        for path_suffix in MODEL_FOLDERS:
            rows = run_experiment_once(path_suffix, seed)
            all_rows.extend(rows)

    if not all_rows:
        print("No results collected. Check your checkpoint folders.")
        return

    df = pd.DataFrame(all_rows)
    aggregate_and_plot(df)

if __name__ == "__main__":
    main()
