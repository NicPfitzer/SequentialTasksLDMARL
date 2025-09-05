import hydra
import os
import json
import re
import time
from pathlib import Path
import pandas as pd

from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd

import torch
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from torchrl.envs.utils import ExplorationType, set_exploration_type
from benchmarl.experiment import Experiment

@torch.no_grad()
def _evaluation_loop(self):
    evaluation_start = time.time()
    with set_exploration_type(
        ExplorationType.DETERMINISTIC
        if self.config.evaluation_deterministic_actions
        else ExplorationType.RANDOM
    ):
        if self.task.has_render(self.test_env) and self.config.render:
            video_frames = []

            def callback(env, td):
                video_frames.append(
                    self.task.__class__.render_callback(self, env, td)
                )

        else:
            video_frames = None
            callback = None

        if self.test_env.batch_size == ():
            rollouts = []
            for eval_episode in range(self.config.evaluation_episodes):
                rollouts.append(
                    self.test_env.rollout(
                        max_steps=self.max_steps,
                        policy=self.policy,
                        callback=callback if eval_episode == 0 else None,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                    )
                )
        else:
            rollouts = self.test_env.rollout(
                max_steps=self.max_steps,
                policy=self.policy,
                callback=callback,
                auto_cast_to_device=True,
                break_when_any_done=False,
                break_when_all_done=True,
                # We are running vectorized evaluation we do not want it to stop when just one env is done
            )
            rollouts = list(rollouts.unbind(0))
    evaluation_time = time.time() - evaluation_start
    self.logger.log(
        {"timers/evaluation_time": evaluation_time}, step=self.n_iters_performed
    )
    self.logger.log_evaluation(
        rollouts,
        video_frames=video_frames,
        step=self.n_iters_performed,
        total_frames=self.total_frames,
    )
    # Callback
    self._on_evaluation_end(rollouts)


NUM_ROLLOUTS = 500
VISIT_MEAN = 25.0

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Interactive evaluation loop that encodes an instruction, runs the BenchMARL
    experiment, and saves a *single* image that contains **both** the occupancy
    heat-map and the team-spread time-series side-by-side.
    """

    # ---------------------------------------------------------------------
    # 1) Static settings that never change inside the loop
    # ---------------------------------------------------------------------
    restore_path = (
        "/Users/nicolaspfitzer/ProrokLab/SequentialTasks/checkpoints/"
        "four_flags/one_color/no_rnn_no_gnn.pt"
    )
    
    experiment_name = list(cfg.keys())[0]
    seed = cfg.seed
    cfg = cfg[experiment_name]  # Get the config for the specific experiment

    cfg.experiment.restore_file = restore_path
    cfg.experiment.restore_map_location = "cpu"
    cfg.experiment.save_folder = Path(os.path.dirname(os.path.realpath(__file__))) / "experiments"
    cfg.experiment.loggers[0] = "csv"
    print(Path(os.path.dirname(os.path.realpath(__file__))) / "experiments")
    cfg.experiment.render = True
    cfg.experiment.evaluation_episodes = NUM_ROLLOUTS
    
    #Experiment._evaluation_loop = _evaluation_loop

    print("Loaded Hydra config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    
    experiment = benchmarl_setup_experiment(cfg, seed=seed, main_experiment=False)
    experiment.evaluate()
    agents = experiment.test_env.base_env._env.scenario.world.agents
    target_flag = experiment.test_env.base_env._env.scenario.switch.target_flag
    hit_switch = torch.zeros((NUM_ROLLOUTS,), device="cpu", dtype=torch.bool)
    found_flags = torch.zeros((NUM_ROLLOUTS,4), device="cpu", dtype=torch.bool)
    on_goal = torch.ones((NUM_ROLLOUTS,), device="cpu", dtype=torch.bool)
    for agent in agents:
        
        hit_switch |= agent.hit_switch
        found_flags |= agent.found_flags
        on_goal &= agent.on_goal

    correct_flag_hit_ratio = found_flags[torch.arange(found_flags.shape[0]), target_flag].float().mean().item()
    switch_hit_ratio = hit_switch.float().mean().item()
    on_goal_ratio = on_goal.float().mean().item()

    print(f"Correct flag hit ratio: {correct_flag_hit_ratio}")
    print(f"Switch hit ratio: {switch_hit_ratio}")
    print(f"On goal ratio: {on_goal_ratio}")

if __name__ == "__main__":
    main()
