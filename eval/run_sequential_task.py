
import hydra
import json
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import torch
from benchmarl.experiment import Experiment


from vmas.simulator.utils import save_video
from sentence_transformers import SentenceTransformer

from pathlib import Path
import os
from trainers.benchmarl_setup_experiment import benchmarl_setup_experiment
import random
NUM_ROLLOUTS = 1

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    
    print("Loaded Hydra config:\n" + OmegaConf.to_yaml(cfg, resolve=True))
    
    # Get the experiment name as first key in the config
    experiment_name = list(cfg.keys())[0]
    #seed = cfg.seed
    restore_path = cfg.restore_path
    cfg[experiment_name].experiment.restore_file = restore_path
    cfg = cfg[experiment_name]  # Get the config for the specific experiment

    # Pre-load the sentence-encoder once
    llm = SentenceTransformer("thenlper/gte-large", device="cpu")
    
    # Prepare deterministic directories relative to project root
    root_dir = get_original_cwd()
    data_dir = os.path.join(root_dir, "data")
    
    # ------------------------------------------------------------------
    # 2) Interactive evaluation loop
    # ------------------------------------------------------------------
    eval_id = 0  # incremental counter for file names
    print("\nEnter instructions to run task (blank / 'quit' to stop).\n")
    
    while True:
        seed = random.randint(0, 10000)  # Use a random seed for each run
        new_sentence = input("Instruction > ").strip()
        if new_sentence.lower() in {"quit", "q", "exit"}:
            print("Exiting evaluation loop.")
            break

        # --------------------------------------------------------------
        # Encode sentence -> embedding (1D tensor)
        # --------------------------------------------------------------
        try:
            if new_sentence == "":
                embedding = torch.zeros(llm.get_sentence_embedding_dimension(), device="cpu")
                print("Using zero embedding for empty instruction.")
            else:
                embedding = torch.tensor(llm.encode([new_sentence]), device="cpu").squeeze(0)
        except Exception as e:
            print(f"Failed to encode instruction: {e}")
            continue

        # --------------------------------------------------------------
        # Serialize to JSON (overwrite each run)
        # --------------------------------------------------------------
        json_path = os.path.join(data_dir, "evaluation_instruction.json")
        payload = {
            #"grid": [0.0] * 100,
            "summary": new_sentence,
            "y": embedding.tolist(),
        }
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(payload, jf)
        print(f"Saved instruction & embedding → {json_path}")

        # Tell the task where the freshly-written JSON lives
        cfg.task.params.data_json_path = json_path
        cfg.experiment.restore_file = restore_path
        experiment = benchmarl_setup_experiment(cfg, seed)
        experiment.evaluate()

if __name__ == "__main__":
    main()