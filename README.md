# SequentialTasks (BenchMARL + VMAS)

This repo runs multi-agent sequential task experiments (training + evaluation) using **BenchMARL** on **VMAS** scenarios, with configs managed by **Hydra**.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- If you want CUDA/GPU, you may need to install `torch` / `torch-geometric` (and `torch_cluster`) using the official wheels for your CUDA version rather than the pinned CPU wheels in `requirements.txt`.

## Training

### Main entrypoint

Run training from the repo root:

```bash
python trainers/benchmarl_train.py
```

This uses Hydra config `conf/config.yaml` by default (see the `defaults:` list inside it).

### Choose what to train

Option A — pick an alternative top-level config:

```bash
python trainers/benchmarl_train.py --config-name=config_four_rooms_rl
python trainers/benchmarl_train.py --config-name=config_four_rooms_naive
python trainers/benchmarl_train.py --config-name=config_four_rooms_rollout
python trainers/benchmarl_train.py --config-name=config_stage_two
```

Option B — edit `conf/config.yaml` and switch which training preset is enabled in `defaults:`.

### Common overrides (Hydra)

Override the global seed:

```bash
python trainers/benchmarl_train.py seed=123
```

Resume/restore from a checkpoint (the training script forwards `restore_path` into `experiment.restore_file`):

```bash
python trainers/benchmarl_train.py --config-name=config_four_rooms_rl \
  restore_path=checkpoints/four_rooms/four_rooms_rl/checkpoint_900000.pt
```

### Outputs, logs, checkpoints

- Hydra creates a run directory under `outputs/` (contains `.hydra/` with the resolved config).
- BenchMARL artifacts (CSV logs, checkpoints, etc.) are written under `trainers/experiments/` by default (see `trainers/benchmarl_setup_experiment.py`; it currently forces `save_folder` to that location).
- Pre-collected checkpoints used by the evaluation scripts live in `checkpoints/` (e.g. `checkpoints/four_rooms/...`).

If you want device changes, adjust the experiment config (example: `conf/*/experiment/*.yaml`) fields like `train_device` / `sampling_device`.

## Evaluation

### Evaluation during training

Most experiment configs enable evaluation (see `conf/*/experiment/*.yaml`), controlled by:
- `evaluation: true`
- `evaluation_interval`
- `evaluation_episodes`
- `evaluation_deterministic_actions`

These evaluations run automatically while training and are logged alongside training metrics (typically via the CSV logger).

### Offline evaluation (post-training)

Scripts in `evaluations/` load Hydra configs, restore a checkpoint, run rollouts, and write summary artifacts under `evaluations/experiments/`.

Single-checkpoint evaluation (recommended starting point):

1) Edit the constants at the top of `evaluations/benchmarl_evaluate_four_rooms_single_checkpoint.py` (at least `CHECKPOINT_PATH` and `CONFIG_NAME`).
2) Run:

```bash
python evaluations/benchmarl_evaluate_four_rooms_single_checkpoint.py
```

Batch evaluation across many checkpoints (Four Rooms):

- Adjust `RESTORE_PATH_ROOT`, `MODEL_FOLDERS`, `NUM_ROLLOUTS`, and seeds at the top of `evaluations/benchmarl_evaluate_four_rooms.py`.
- Run:

```bash
python evaluations/benchmarl_evaluate_four_rooms.py
```

Four Flags evaluation:

- `evaluations/benchmarl_evaluate_four_flags.py` is an example evaluation loop, but it currently contains a hard-coded `restore_path` that you’ll likely want to change to a repo-relative path under `checkpoints/`.

### Interactive evaluation (text instruction → rollout)

`eval/run_sequential_task.py` is intended for interactive rollouts driven by typed instructions and a sentence encoder.

Example:

```bash
python eval/run_sequential_task.py --config-name=config_four_flags_rl \
  restore_path=checkpoints/four_flags/one_color/policy.pt
```

It writes the current instruction embedding to `data/evaluation_instruction.json` each run.

## Troubleshooting

- **Hydra changed my working directory**: expected; outputs go under `outputs/`. Use absolute paths or `hydra.run.dir=.` if you need to keep the CWD stable.
- **No checkpoints found**: confirm the experiment’s `checkpoint_interval` and where your run writes artifacts (`trainers/experiments/` vs `checkpoints/`).
- **Import / wheel issues (torch-geometric / torch_cluster)**: install the correct wheels for your Python/CUDA combo, then reinstall the remaining requirements.
