import torch
from benchmarl.experiment import Experiment
from collections import deque, OrderedDict
from typing import Any, Dict, List, Optional

class ExperimentFix(Experiment):
    
    def __init__(self, task, algorithm_config, model_config, seed, config, critic_model_config = None, callbacks = None):
        super().__init__(task, algorithm_config, model_config, seed, config, critic_model_config, callbacks)
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """Load the state_dict for the experiment.

        Args:
            state_dict (dict): the state dict

        """
        for group in self.group_map.keys():
            self.losses[group].load_state_dict(state_dict[f"loss_{group}"])
        if not self.config.collect_with_grad:
            self.collector.load_state_dict(state_dict["collector"])
        self.total_time = state_dict["state"]["total_time"]
        self.total_frames = state_dict["state"]["total_frames"]
        self.n_iters_performed = state_dict["state"]["n_iters_performed"]
        self.mean_return = state_dict["state"]["mean_return"]
