import os
import re
from pathlib import Path
import pandas as pd  # not strictly needed, but handy if you print timestamps, etc.
import torch

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment
from benchmarl.experiment import Experiment

# ------------- user settings -------------
CHECKPOINT_PATH = "checkpoints/four_rooms/four_rooms_rl/checkpoint_1950000.pt"  # set this
CONFIG_PATH = "../conf"                                     # hydra config folder
CONFIG_NAME = "config_four_rooms_rl"                        # e.g. config_four_rooms_rl | config_four_rooms_naive | config_four_rooms_rollout
DEVICE = "cpu"
EVAL_EPISODES = 500
SEED = 21
# -----------------------------------------

def generate_cfg(
    config_path: str,
    config_name: str,
    restore_path: str,
    device: str = "cpu",
) -> tuple[DictConfig, int]:
    """Build hydra config for a single run, injecting restore path and device mapping."""
    if not os.path.isabs(restore_path):
        restore_path = os.path.abspath(restore_path)
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=config_name, overrides=[])
    experiment_name = list(cfg.keys())[0]
    seed = SEED

    # Special case if your rollout config expects a base policy file and a separate policy restore
    if config_name == "config_four_rooms_rollout":
        # Use the checkpoint as the policy restore
        cfg[experiment_name].task.params.policy_restore_path = restore_path
        # And point experiment.restore_file to the base policy if your config expects it
        # Change the path below if your project stores it elsewhere
        cfg[experiment_name].experiment.restore_file = "checkpoints/four_rooms/four_rooms_rollout/policy_base.pt"
    else:
        cfg[experiment_name].experiment.restore_file = restore_path

    cfg[experiment_name].experiment.restore_map_location = device
    cfg = cfg[experiment_name]
    return cfg, seed

def main():
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    cfg, seed = generate_cfg(
        config_path=CONFIG_PATH,
        config_name=CONFIG_NAME,
        restore_path=CHECKPOINT_PATH,
        device=DEVICE,
    )

    # keep it minimal, no logging to disk
    if "loggers" in cfg.experiment and isinstance(cfg.experiment.loggers, list):
        cfg.experiment.loggers = []  # disable loggers

    cfg.experiment.render = False
    cfg.experiment.evaluation_episodes = EVAL_EPISODES

    experiment: Experiment = benchmarl_setup_experiment(cfg, seed=seed, main_experiment=False)
    experiment.evaluate()

    # Aggregate metrics over agents per episode
    agents = experiment.test_env.base_env._env.scenario.world.agents
    on_goal = torch.ones((EVAL_EPISODES,), device="cpu", dtype=torch.bool)
    opened_final_room = torch.zeros((EVAL_EPISODES,), device="cpu", dtype=torch.bool)
    for agent in agents:
        # expect per-episode boolean tensors
        on_goal &= agent.on_goal
        # last switch indicates final room opened, tweak indexing if your env differs
        opened_final_room |= agent.switch_hits[:, -1].bool()

    on_goal_ratio = on_goal.float().mean().item()
    opened_final_room_ratio = opened_final_room.float().mean().item()

    # Print results
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Episodes:   {EVAL_EPISODES}")
    print(f"On-goal ratio:             {on_goal_ratio:.3f}")
    print(f"Opened final room ratio:   {opened_final_room_ratio:.3f}")

if __name__ == "__main__":
    main()
