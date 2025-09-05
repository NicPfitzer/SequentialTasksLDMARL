import os
import re
from pathlib import Path


from omegaconf import DictConfig

import torch
from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment
from benchmarl.experiment import Experiment

import matplotlib.pyplot as plt
import pandas as pd

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


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
    "four_rooms_rollout/",
]

CONFIGS = {
    "leftmost": {"initial_room": 0, "even_distribution": False},
    "room 1": {"initial_room": 1, "even_distribution": False},
    "room 2": {"initial_room": 2, "even_distribution": False},
    "rightmost": {"initial_room": 3, "even_distribution": False},
    "random": {"initial_room": None, "even_distribution": False},
    "even": {"initial_room": None, "even_distribution": True},
}

# Output
OUT_ROOT = Path(os.path.dirname(os.path.realpath(__file__))) / "experiments" / "four_rooms_initialization"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_ROOT / "summary_seeds.csv"
PLOT_GOAL = OUT_ROOT / "on_goal_mean_ci.png"
PLOT_OFR  = OUT_ROOT / "opened_final_room_mean_ci.png"
PLOT_SUCCESS = OUT_ROOT / "success_level_mean_ci.png"


# --------------------------- helpers ---------------------------------
def pretty_model_label(path_suffix: str, idx: int) -> str:
    name = path_suffix.strip("/").lower()
    if "naive" in name:
        return "vanilla_rl"
    if "multitask" in name or "rollout" in name:
        # keep "rollout" label if you prefer
        return f"{'rollout' if 'rollout' in name else 'multitask'}_{idx}"
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

def run_experiment_once(path_suffix: str, seed: int, config: dict, config_type: str, model_idx: int) -> list[dict]:

    rows = []
    
    if path_suffix == "four_rooms_rollout/":
        ckpt_dir = os.path.join(RESTORE_PATH_ROOT, "four_rooms_multitask/")
    else:
        ckpt_dir = os.path.join(RESTORE_PATH_ROOT, path_suffix)

    if not os.path.isdir(ckpt_dir):
        print(f"[warn] Missing directory: {ckpt_dir}")
        return rows

    restore_path = os.path.join(ckpt_dir, "policy.pt")

    cfg = generate_cfg(
        config_path="../conf",
        config_name="config_" + path_suffix[:-1],  # strip trailing slash
        restore_path=restore_path,
        device=DEVICE,
        seed=seed,
    )
    
    model_label = pretty_model_label(path_suffix, model_idx)
    
    try:
        cfg.experiment.loggers[0] = "csv"
    except Exception:
        pass
    cfg.experiment.render = False
    cfg.experiment.evaluation_episodes = NUM_ROLLOUTS
    cfg.experiment.save_folder = OUT_ROOT / f"{model_label}_seed{seed}"
    
    cfg.task.params.even_distribution = config.get("even_distribution", False)
    cfg.task.params.initial_room = config.get("initial_room", None)
    if path_suffix in "four_rooms_rollout/" and model_idx == 3:
        cfg.task.params.initialized_rnn = True

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
            "config_type": config_type,
        }
    )
    return rows
    
    
def aggregate_and_plot(df: pd.DataFrame) -> None:
    """Pool across seeds and plot grouped error bars per config for success_level."""
    # Save raw rows
    df.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved per-seed CSV → {SUMMARY_CSV}")

    # Pretty labels
    label_map = {
        "vanilla_rl": "Vanilla RL",
        "naive_rl": "Vanilla RL",
        "rl": "Tuned Rewards RL",
        "rollout_1": "DeCLaRE - No Language Init.",
        "rollout_2": "DeCLaRE - Language Init.",
    }
    df = df.copy()
    df["model"] = df["model"].map(lambda m: label_map.get(m, m))

    # Desired x-axis order (only keep those present)
    desired_order = ["leftmost", "room 1", "room 2", "rightmost", "random", "even"]
    present = list(df["config_type"].unique())
    config_order = [c for c in desired_order if c in present] or sorted(present)

    # Consistent model order
    model_order = [m for m in ["Vanilla RL", "Tuned Rewards RL", "DeCLaRE - No Language Init.", "DeCLaRE - Language Init."]
                   if m in set(df["model"])]

    # Pooled mean/std across seeds given per-seed mean/std and n (episodes)
    def pooled_mean_std(group: pd.DataFrame, mean_col: str, std_col: str, n_col: str):
        means = group[mean_col].to_numpy()
        stds  = group[std_col].to_numpy()
        ns    = group[n_col].to_numpy().astype(float)
        N = ns.sum()
        if N <= 1 or len(means) == 0:
            M = means.mean() if len(means) else 0.0
            return pd.Series({"pooled_mean": float(M), "pooled_std": 0.0, "N": int(N)})
        M = (means * ns).sum() / N
        ssw = ((ns - 1.0) * (stds ** 2)).sum()        # within
        ssb = (ns * ((means - M) ** 2)).sum()         # between
        var = (ssw + ssb) / max(N - 1.0, 1.0)
        return pd.Series({"pooled_mean": float(M), "pooled_std": float(var ** 0.5), "N": int(N)})

    # Aggregate once: pooled stats per (config, model)
    pooled = (
        df.groupby(["config_type", "model"], as_index=False)
          .apply(lambda g: pooled_mean_std(g, "success_level_mean", "success_level_std", "episodes"))
          .reset_index(drop=True)
    )

    # Plot grouped error bars
    import numpy as np
    x = np.arange(len(config_order), dtype=float)
    n_models = len(model_order)
    if n_models == 0 or len(config_order) == 0:
        print("[warn] Nothing to plot for Success.")
        return

    group_width = 0.60
    step = group_width / max(n_models, 1)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * step

    plt.figure(figsize=(1.4 * len(config_order) + 2.5, 5.2))
    cap = 4

    for i, model in enumerate(model_order):
        sub = pooled[pooled["model"] == model].set_index("config_type")
        means = [sub.loc[c, "pooled_mean"] if c in sub.index else np.nan for c in config_order]
        stds  = [sub.loc[c, "pooled_std"]  if c in sub.index else np.nan for c in config_order]
        xi = x + offsets[i]
        plt.errorbar(
            xi, means, yerr=stds,
            fmt="o", capsize=cap, elinewidth=1.2, linewidth=0.0,  # markers only, no lines
            label=model
        )

    plt.xticks(x, config_order)
    plt.xlabel("Config")
    plt.ylabel("Success (mean ± std)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(title="Model", loc="best")
    plt.tight_layout()
    plt.savefig(PLOT_SUCCESS, dpi=150)
    plt.savefig(PLOT_SUCCESS.with_suffix(".pdf"))
    plt.close()
    print(f"Saved plot → {PLOT_SUCCESS}")

# --------------------------- main ---------------------------------
def main():
    print(f"Seeds: {SEEDS}")
    all_rows = []
    for seed in SEEDS:
        for i, path_suffix in enumerate(MODEL_FOLDERS):
            for key, config in CONFIGS.items():
                rows = run_experiment_once(path_suffix, seed, config, config_type=key, model_idx=i)
                all_rows.extend(rows)
    if not all_rows:
        print("No results collected. Check your checkpoint folders.")
        return

    df = pd.DataFrame(all_rows)
    aggregate_and_plot(df)

if __name__ == "__main__":
    main()
