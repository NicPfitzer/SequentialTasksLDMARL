import hydra
from omegaconf import DictConfig, OmegaConf
from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment
#from trainers.benchmarl_setup_experiment_new import benchmarl_setup_experiment


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))   # full merged config
    
    # Get the experiment name as first key in the config
    experiment_name = list(cfg.keys())[0]
    seed = cfg.seed
    if "restore_path" in cfg:
        restore_path = cfg.restore_path
        cfg[experiment_name].experiment.restore_file = restore_path
    cfg = cfg[experiment_name]  # Get the config for the specific experiment

    experiment = benchmarl_setup_experiment(cfg, seed=seed, main_experiment=True)
    experiment.run()

if __name__ == "__main__":
    main()