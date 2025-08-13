from omegaconf import DictConfig
from pathlib import Path
from benchmarl.environments import VmasTask
from torchrl.envs import VmasEnv
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments.common import Task
from vmas import scenarios
# from trainers.experiment_fix import ExperimentFix
from benchmarl.algorithms import MappoConfig
from hydra.utils import instantiate
from utils.utils import _load_class
import importlib
import copy

def _patch_env_creator():
    """Monkey-patch VmasTask.get_env_fun so we can pass a dotted class path."""

    def get_env_fun(self, num_envs, continuous_actions, seed, device):
        # clone the config dict to avoid side-effects
        cfg = dict(copy.deepcopy(self.config))
        
        return lambda: VmasEnv(
            scenario= _load_class(self.scenario_class)(),  # instantiate custom scenario
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **cfg,
        )

    VmasTask.get_env_fun = get_env_fun

def benchmarl_setup_experiment(cfg: DictConfig, seed: int, main_experiment: bool) -> Experiment:

    # ---------- TASK ----------
    task_enum = VmasTask[cfg.task.name]
    task_enum.config = None
    task = task_enum.get_from_yaml()
    scenario_class_name = cfg.task.scenario_class
    _ , scenario_name  = scenario_class_name.rsplit(".", 1)
    task.config = cfg.task.params
    #task.name = scenario_name
    task.scenario_class = scenario_class_name
    
    _patch_env_creator()

    # ---------- ALGORITHM ----------
    if cfg.algorithm.type.lower() == "mappo":
        algorithm_config = MappoConfig.get_from_yaml()
    else:
        raise ValueError("Only MAPPO implemented here")

    algorithm_config.entropy_coef = cfg.algorithm.params.entropy_coef

    # ---------- ACTOR MODEL ----------
    actor_model = instantiate(cfg.model.actor_model)
    actor_model.activation_class = _load_class(actor_model.activation_class)
    actor_model.layer_class = _load_class(actor_model.layer_class)

    if getattr(actor_model, "use_gnn", False) or getattr(actor_model, "use_event_gnn", False):
        actor_model.gnn_class = _load_class(actor_model.gnn_class)
    else:
        actor_model.gnn_class = None

    # ---------- CRITIC MODEL ----------
    critic_model = instantiate(cfg.model.critic_model)
    critic_model.activation_class = _load_class(critic_model.activation_class)
    critic_model.layer_class = _load_class(critic_model.layer_class)
    if getattr(critic_model, "use_gnn", False) or getattr(critic_model, "use_event_gnn", False):
        critic_model.gnn_class = _load_class(critic_model.gnn_class)
    else:
        critic_model.gnn_class = None

    # ---------- EXPERIMENT CONFIG ----------
    exp_cfg = ExperimentConfig(**cfg.experiment)
    exp_cfg.save_folder = Path(__file__).parent / "experiments"
    exp_cfg.save_folder.mkdir(exist_ok=True, parents=True)

    # ---------- RETURN EXPERIMENT ----------

    return Experiment(
        task=task,
        algorithm_config=algorithm_config,
        model_config=actor_model,
        critic_model_config=critic_model,
        seed=seed,
        config=exp_cfg,
    )

