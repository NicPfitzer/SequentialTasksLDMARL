
"""
Launch several BenchMARL experiments sequentially.

"""

from pathlib import Path
from omegaconf import OmegaConf
import hydra
from hydra import compose, initialize

from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment


# ---------------------------------------------------------------------------
# 1. List every training variant you want to run.
#    Each item is a perfectly normal Hydra override string.
#    Feel free to add / remove entries or build this list programmatically.
# ---------------------------------------------------------------------------
TRAINING_VARIANTS = [
    "+four_flags/training=four_flags_multitask",  # maps to conf/four_flags/training/four_flags_multitask.yaml
    "+four_flags/training=four_flags_naive",
    "+four_flags/training=four_flags_rl",
]

# Optional: choose where Hydra puts its output directories.
# Using ${hydra.job.name} keeps them distinct for each variant.
RUN_DIR_TEMPLATE = "outputs/${hydra.job.name}/${now:%Y-%m-%d_%H-%M-%S}"


def run_all():
    """Compose, build and run every experiment in TRAINING_VARIANTS sequentially."""
    conf_root = "../conf"   # ../conf relative to this script

    for override in TRAINING_VARIANTS:
        print(f"\n=== Launching experiment with override: {override} ===")

        # Every loop starts a fresh Hydra context so global state is reset.
        with initialize(config_path=conf_root, version_base=None):
            # Build the full config, applying our override *and* a unique run dir.
            cfg = compose(
                config_name="config",
                overrides=[
                    override,
                    f"hydra.run.dir={RUN_DIR_TEMPLATE}",          # put logs in a unique folder
                    f"hydra.job.name={override.split('=')[1]}",   # nicer folder names
                ],
            )

            print("-" * 60)
            print(OmegaConf.to_yaml(cfg, resolve=True))
            print("-" * 60)

            experiment_name = list(cfg.keys())[0]
            seed = cfg.seed
            if "restore_path" in cfg:
                restore_path = cfg.restore_path
                cfg[experiment_name].experiment.restore_file = restore_path
            cfg = cfg[experiment_name]  # Get the config for the specific experiment

            experiment = benchmarl_setup_experiment(cfg, seed=seed, main_experiment=True)
            experiment.run()           # blocking – when this returns the next job starts

        print(f"=== Finished: {override} ===\n")


if __name__ == "__main__":
    run_all()
