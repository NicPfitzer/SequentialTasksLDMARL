import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths of the uploaded files
path_multitask = "/Users/nicolaspfitzer/ProrokLab/SequentialTasks/eval/multitask_policy.csv"
path_full_rl = "/Users/nicolaspfitzer/ProrokLab/SequentialTasks/eval/full_rl_policy.csv"

def load_reward_series(csv_path):
    """
    Load a reward (or return) series from a CSV file.
    Heuristic:
    1. Prefer a column that contains the word 'reward' (case-insensitive).
    2. Otherwise, look for a column that contains 'return'.
    3. If neither, pick the last numeric column.
    Returns a pandas Series.
    """
    df = pd.read_csv(csv_path)
    
    # Identify reward-like columns
    reward_cols = [c for c in df.columns if 'reward' in c.lower()]
    if reward_cols:
        col = reward_cols[0]
    else:
        return_cols = [c for c in df.columns if 'return' in c.lower()]
        if return_cols:
            col = return_cols[0]
        else:
            # fallback: pick last numeric column
            numeric_cols = df.select_dtypes(include='number').columns
            if numeric_cols.empty:
                raise ValueError(f"No numeric columns found in {csv_path}")
            col = numeric_cols[-1]
    
    return df[col]

# Load the reward series
reward_multitask = load_reward_series(path_multitask)
reward_full_rl = load_reward_series(path_full_rl)

# Plot for multitask_policy
plt.figure()
plt.plot(reward_multitask.reset_index(drop=True))
plt.title("Reward over episodes – Multitask Policy")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot for full_rl_policy
plt.figure()
plt.plot(reward_full_rl.reset_index(drop=True))
plt.title("Reward over episodes – Full RL Policy")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid(True)
plt.tight_layout()
plt.show()
