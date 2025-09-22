# rnn_systematic_eval_with_plots.py
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",    # use serif fonts
    "mathtext.fontset": "cm",  # Computer Modern for math
    "font.serif": ["cmr10"],   # Computer Modern Roman
})
from sentence_transformers import SentenceTransformer
import textwrap

from sequence_models.multitask.model_training.rnn_model import EventRNN
from sequence_models.multitask.random_automaton_walk_multitask import (
    FIND_RED, FIND_GREEN, FIND_BLUE, FIND_PURPLE, FIND_SWITCH, FIND_GOAL,
    EXPLORE, NAV_HOME, DEFEND_WIDE, DEFEND_TIGHT, IDLE,
    FIND_FIRST_SWITCH, FIND_SECOND_SWITCH, FIND_THIRD_SWITCH
)

NUM_AUTOMATA = 3  # number of tasks (first N bits in state)
DELTA = 1e-6
# -----------------------
# Config and dictionaries
# -----------------------
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_CKPT = "sequence_models/multitask/multitask.pth"
CSV_OUT = "rnn_eval_results.csv"
PLOT_DIR = "plots"

task_map = {
    0: "Locate the red flag, then the switch, and advance to the goal.",
    1: "Search for the green flag, then the switch, and navigate to the goal.",
    2: "Find the blue flag, spot the switch, and head for the target.",
    3: "Identify the purple flag, navigate to the switch and proceed to the goal.",
    4: "Locate the mission flag and defend the position; adapt your defense.",
    5: "Find the objective flag, and then deliver it back to the starting zone using shortest paths.",
    6: "Recon agents, unlock the first, second, and third switches in sequence, then advance to the objective room and reach the target."
}

subtask_map = {
    FIND_RED: "Find Red Flag",
    FIND_GREEN: "Find Green Flag",
    FIND_BLUE: "Find Blue Flag",
    FIND_PURPLE: "Find Purple Flag",
    FIND_SWITCH: "Navigate to the Switch",
    FIND_GOAL: "Navigate to the Goal",
    EXPLORE: "Explore",
    DEFEND_WIDE: "Defend Wide",
    DEFEND_TIGHT: "Defend Tight",
    NAV_HOME: "Navigate Home",
    IDLE: "Idle",
    FIND_FIRST_SWITCH: "Find First Switch",
    FIND_SECOND_SWITCH: "Find Second Switch",
    FIND_THIRD_SWITCH: "Find Third Switch",

}
SUBTASK_TO_IDX = {name: i for i, name in subtask_map.items()}

# Put near other globals
EVENT_ORDER = ["Red", "Green", "Blue", "Purple", "Switch", "Found Flag", "Spotted Enemy", "Found Base", "Found First Switch", "Found Second Switch", "Found Third Switch"]
REVERSE_TASK_MAP = {v: k for k, v in task_map.items()}  # text -> task_id

# def event_vec_to_text(vec):
#     # vec is a list of 8 ints/floats [R,G,B,P,S,F,SE,FB]
#     names = [EVENT_ORDER[i] for i, v in enumerate(vec) if int(v) == 1]
#     return "+".join(names) if names else "None"

def event_vec_to_text(vec):
    # if bits are all zero => None
    if int(np.sum(vec)) == 0:
        return "No Event"
    # vec is one-hot
    return EVENT_ORDER[int(np.argmax(vec))]

TASK_COLOR_ORDER = ["red", "green", "blue", "purple", "brown", "black", "orange"]
TASK_COLORS = {tid: TASK_COLOR_ORDER[tid % len(TASK_COLOR_ORDER)] for tid in sorted(task_map.keys())}

def task_color(tid: int):
    return TASK_COLORS.get(tid, "black")

def marker_facecolors_for_states(states, color, fill_states=("Navigate to the Goal", "Idle")):
    """Return a list of facecolors for each state string.
    Fill with the given task color if in fill_states, otherwise white.
    """
    return [color if s in fill_states else "white" for s in states]

from matplotlib.transforms import offset_copy
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch

def draw_stationary_loops(ax, x, y, color="black", side=1, size=13, outline=None,
                          xspan=0.23, yoff=0.08, rad=-0.9):
    """
    Draw a small curved arrow under points where y[i-1] == y[i].
    - Curves downward (rad < 0).
    - Arrowhead on the right if side > 0, left if side < 0.
    """
    for i in range(1, len(x)):
        if np.isclose(y[i-1], y[i]):
            xc, yc = float(x[i]), float(y[i])
            yb = yc - yoff      # drop the loop below the marker
            w  = xspan * 0.5

            if side <= 0:       # head on the right
                x0, y0 = xc - w, yb
                x1, y1 = xc + w, yb
            else:               # head on the left
                x0, y0 = xc + w, yb
                x1, y1 = xc - w, yb

            arrow = FancyArrowPatch(
                (x0, y0), (x1, y1),
                connectionstyle=f"arc3,rad={rad}",  # rad < 0 makes it bow downward
                arrowstyle="-|>",
                mutation_scale=6,
                lw=1.5,
                color=color,
                zorder=15,
                path_effects=outline
            )
            ax.add_patch(arrow)

# def draw_stationary_loops(ax, x, y, color="black", side=1, size=13, outline=None,
#                           dx=-0, dy=12):
#     """
#     Draw a small loop arrow at marker i when y[i-1] == y[i].
#     Offset controlled by dx, dy (points).
#     """
#     transform = offset_copy(ax.transData, fig=ax.figure, x=dx*side, y=-dy, units="points")

#     for i in range(1, len(x)):
#         if np.isclose(y[i-1], y[i]):
#             ax.text(
#                 x[i], y[i], r"$\circlearrowright$",
#                 transform=transform, ha="center", va="center",
#                 fontsize=size, color=color, zorder=15,
#                 path_effects=outline if outline is not None else None,
#             )
# -----------------------
# Model loading
# -----------------------
def load_event_rnn_model(checkpoint_path: str = MODEL_CKPT) -> EventRNN:
    model = EventRNN(
        event_dim=11, y_dim=1024, latent_dim=1024, state_dim=17, input_dim=64, num_layers=1
    )
    state = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

# -----------------------
# Inference helper
# -----------------------
@torch.no_grad()
def predict_state_rollout(task: str, subtask: str | None, events: list[list[float]],
                          model: EventRNN, embedder: SentenceTransformer) -> dict:
    y = embedder.encode(task, convert_to_tensor=True).unsqueeze(0)  # (1, y_dim)
    h = embedder.encode(subtask, convert_to_tensor=True).unsqueeze(0) if subtask else torch.zeros_like(y)

    T = len(events)
    e = torch.tensor(events, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, e_dim)
    y = y.unsqueeze(1).expand(-1, T, -1).to(DEVICE)                        # (1, T, y_dim)

    if subtask is None:
        state_logits, _ = model.rollout(e, y, h)  # (1, T, state_dim)
    else:
        state_logits, _ = model._initalized_rollout(e, y, h)
    probs = torch.sigmoid(state_logits).squeeze(0)        # (T, state_dim)
    preds = (probs > 0.5).int()                           # (T, state_dim)

    # Task index from first NUM_AUTOMATA bits
    binarized = preds[:, :NUM_AUTOMATA]
    weights = 2 ** torch.arange(NUM_AUTOMATA - 1, -1, -1, device=binarized.device)
    task_index = (binarized * weights).sum(dim=1).int().tolist()
    tasks = [task_map.get(i, f"Unknown({i})") for i in task_index]

    # Subtask index from the remaining bits (argmax)
    subtask_index = probs[:, NUM_AUTOMATA:].argmax(dim=-1).tolist()
    subtasks = [subtask_map.get(i, f"Unknown({i})") for i in subtask_index]

    return {
        "pred_probs": probs.cpu().tolist(),
        "pred_labels": preds.cpu().tolist(),
        "tasks": tasks,
        "subtasks": subtasks,
    }


# -----------------------
# Sequences and experiments
# -----------------------
def build_sequences():
    # Events: [R, G, B, P, S, FoundFlag, SpottedEnemy, FoundBase]
    return {
        "S1": [
            [0,0,0,0,0,0,0,0,0,0,0],  # None
            [1+DELTA,0,0,0,0,0,0,0,0,0,0],  # Red
            [1,0,0,0,0,1+DELTA,0,0,0,0,0],  # FoundFlag
            [1,0,0,0,0,1,1+DELTA,0,0,0,0],  # SpottedEnemy
            [1,0,0,0,0,0,1,1+DELTA,0,0,0],  # FoundBase
            [1,0,0,0,0,0,1,0,1+DELTA,0,0],  # Found First switch
            [1,0,1+DELTA,0,0,0,1,0,0,0,0],  # Blue
            [1,0,1,0,1+DELTA,0,1,0,0,0,0],  # Switch
            [0,0,0,0,0,0,0,0,0,0,0],  # None
            [0,0,0,0,0,0,0,0,0,0,0],  # None
            [0,0,0,0,0,0,0,0,0,0,0],  # None
            [0,0,0,0,0,0,0,0,0,0,0],  # None
            [0,0,0,0,0,0,0,0,0,0,0],  # None
            [0,1+DELTA,0,0,0,0,0,0,0,0,0],  # Green
            [0,1,0,1+DELTA,0,0,0,0,0,0,0],  # Purple
            [0,1,0,1,0,0,1,0,1,1+DELTA,0],  # Found Second switch
            [0,1,0,1,0,0,1,0,1,1,1+DELTA],  # Found Third switch
            [0,1,1,0,1+DELTA,0,0,0,0,0,0],  # Switch
            [0,0,0,0,0,0,0,0,0,0,0],  # None
            [0,0,0,0,0,0,0,0,0,0,0],  # None
            [0,0,0,0,0,0,0,0,0,0,0],  # None
        ],
        "S2": [
            [0,1,0,0,0,0,0,0,0,0,0],  # Green
            [0,0,1,0,0,0,0,0,0,0,0],  # Blue
            [0,0,0,0,1,0,0,0,0,0,0],  # Switch
            [0,0,0,1,0,0,0,0,0,0,0],  # Purple
            [1,0,0,0,0,0,0,0,0,0,0],  # Red
            [0,0,0,0,1,0,0,0,0,0,0],  # Switch
        ],
        "S3": [
            [0,0,0,1,0,0,0,0,0,0,0],  # Purple
            [0,0,0,0,1,0,0,0,0,0,0],  # Switch
            [0,0,1,0,0,0,0,0,0,0,0],  # Blue
            [0,0,0,0,1,0,0,0,0,0,0],  # Switch
            [0,1,0,0,0,0,0,0,0,0,0],  # Green
            [1,0,0,0,0,0,0,0,0,0,0],  # Red
            [0,0,0,0,1,0,0,0,0,0,0],  # Switch
        ],
    }

def build_experiments():
    sequences = build_sequences()
    experiments = []
    for task_id, task_text in task_map.items():
        for seq_id, events in sequences.items():
            experiments.append({
                "exp_id": f"T{task_id}_{seq_id}",
                "task_id": task_id,
                "task": task_text,
                "seq_id": seq_id,
                "subtask": None,   # keep consistent across tasks
                "events": events,
            })
    return experiments

# -----------------------
# Runner
# -----------------------
def run_and_save():
    model = load_event_rnn_model(MODEL_CKPT)
    embedder = SentenceTransformer("thenlper/gte-large")

    rows = []
    for spec in build_experiments():
        res = predict_state_rollout(
            task=spec["task"],
            subtask=spec["subtask"],
            events=spec["events"],
            model=model,
            embedder=embedder,
        )
        T = len(spec["events"])
        for t in range(T):
            rows.append({
                "exp_id": spec["exp_id"],
                "task_id": spec["task_id"],
                "seq_id": spec["seq_id"],
                "timestep": t,
                "task_input": spec["task"],
                "subtask_input": spec["subtask"] if spec["subtask"] else "",
                "event": json.dumps(spec["events"][t]),
                "pred_task": res["tasks"][t],
                "pred_subtask": res["subtasks"][t],
                "pred_labels": json.dumps(res["pred_labels"][t]),
                "pred_probs": json.dumps(np.round(res["pred_probs"][t], 6).tolist()),
            })

    df = pd.DataFrame(rows)
    df["pred_task_correct"] = (df["pred_task"] == df["task_input"]).astype(int)
    df.to_csv(CSV_OUT, index=False)
    print(f"Saved {len(df)} rows to {CSV_OUT}")
    return df

# -----------------------
# Plotting
# -----------------------
import os
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_task_accuracy(df: pd.DataFrame):
    # Accuracy per task for each sequence
    for seq_id, g in df.groupby("seq_id"):
        acc = g.groupby("task_id")["pred_task_correct"].mean().reindex(range(6))
        fig = plt.figure(figsize=(10, 4.5))
        plt.bar([f"T{t}" for t in acc.index], acc.values)
        plt.ylim(0, 1)
        plt.ylabel("Task prediction accuracy")
        plt.title(f"Task accuracy by task (sequence {seq_id})")
        for i, v in enumerate(acc.values):
            plt.text(i, v + 0.02, f"{v*100:.1f}%", ha="center", va="bottom")
        out_png = os.path.join(PLOT_DIR, f"task_accuracy_{seq_id}.png")
        out_pdf = os.path.join(PLOT_DIR, f"task_accuracy_{seq_id}.pdf")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.savefig(out_pdf)
        plt.close(fig)

def plot_subtask_timelines(df: pd.DataFrame, add_sentences: bool = False):
    # For each sequence, make a 2x3 grid of subtask timelines (one per task)
    for seq_id, g in df.groupby("seq_id"):
        fig, axes = plt.subplots(2, 3, figsize=(40, 7), sharex=False, sharey=True)
        axes = axes.flatten()
        for task_id in range(6):
            ax = axes[task_id]
            gg = g[g["task_id"] == task_id].copy()
            # Convert subtask names to indices for plotting
            y_vals = gg["pred_subtask"].map(SUBTASK_TO_IDX)
            ax.step(gg["timestep"], y_vals, where="post", marker="o")
            if add_sentences:
                ax.set_title(f"T{task_id}: {task_map.get(task_id, '')}", fontsize=10)
            else:
                ax.set_title(f"Task T{task_id}")
            ax.set_yticks(range(len(subtask_map)))
            ax.set_yticklabels([subtask_map.get(i, "") for i in range(len(subtask_map))], fontsize=18)
            ax.set_xlabel("t")
        if add_sentences:
            fig.suptitle(f"Predicted subtask timelines (sequence {seq_id})", y=0.98, fontsize=18)
        else:
            fig.suptitle(f"Predicted subtask timelines (sequence {seq_id})", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_png = os.path.join(PLOT_DIR, f"subtask_timelines_{seq_id}.png")
        out_pdf = os.path.join(PLOT_DIR, f"subtask_timelines_{seq_id}.pdf")
        plt.savefig(out_png, dpi=200)
        plt.savefig(out_pdf)
        plt.close(fig)


def plot_subtask_progression(df, out_dir="plots", add_sentences: bool = False):
    import os, json, numpy as np, textwrap
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    os.makedirs(out_dir, exist_ok=True)
    wrapper = textwrap.TextWrapper(width=78)

    for seq_id, g in df.groupby("seq_id"):
        g = g.sort_values(["task_id", "timestep"])

        # x-axis (shared, integer timesteps) and labels
        g0 = g[g["task_id"] == g["task_id"].min()].sort_values("timestep")
        x_int = g0["timestep"].to_numpy()
        x_labels = [
            event_vec_to_text(json.loads(e)) if "event_name" not in g0
            else g0["event_name"].tolist()[i]
            for i, e in enumerate(g0["event"].tolist())
        ]

        # layout: top chart, bottom descriptions
        fig = plt.figure(figsize=(35, 8.0), constrained_layout=True)
        if add_sentences:
            gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[3.5, .8], hspace=0.05)
            ax = fig.add_subplot(gs[0, 0])
            ax_desc = fig.add_subplot(gs[1, 0]); ax_desc.axis("off")
        else:
            gs = fig.add_gridspec(nrows=1, ncols=1)
            ax = fig.add_subplot(gs[0, 0])

        # compute per-task jitter
        task_ids = sorted(g["task_id"].unique())
        k = len(task_ids)
        # compute per-task jitter (kept small so categories stay readable)
        x_jitters = {tid: np.linspace(-0.0, 0.0, k)[-i-1] for i, tid in enumerate(task_ids)}
        k = 0
        x_jitters = {}
        x_jitters[0] = -0.054 * k
        x_jitters[1] = -0.00 * k
        x_jitters[2] = 0.054 * k
        x_jitters[3] = 0.00 * k
        x_jitters[4] = -0.054 * k
        x_jitters[5] = 0.054 * k
        x_jitters[6] = 0.00 * k
        # symmetric y jitters to separate identical categorical states
        y_jitters = {tid: (i - (k - 1) / 2) * 0.08 for i, tid in enumerate(task_ids)}
        k = 5.5
        y_jitters = {}
        y_jitters[0] = -0.054 * k
        y_jitters[1] = -0.027 * k
        y_jitters[2] = 0.054 * k
        y_jitters[3] = 0.027 * k
        y_jitters[4] = -0.054 * k
        y_jitters[5] = 0.054 * k
        y_jitters[6] = 0.00 * k
        outline = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]

        # plot each task
        marker_buffer = []
        loop_buffer = []   # NEW: store series for loops

        for i, tid in enumerate(task_ids):
            gg = g[g["task_id"] == tid].sort_values("timestep")
            x = gg["timestep"].to_numpy(dtype=float) + x_jitters[tid]
            y = gg["pred_subtask"].map(lambda s: SUBTASK_TO_IDX.get(s, -1)).to_numpy(dtype=float) + y_jitters[tid]
            c = task_color(tid)

            # draw only the line now
            ax.plot(
                x, y,
                drawstyle="steps-post", linewidth=2.5,
                color=c, label=f"T{tid}",
                zorder=2, path_effects=outline
            )

            states = gg["pred_subtask"].tolist()
            #facecols = marker_facecolors_for_states(states, c)  # pass task color
            states = gg["pred_subtask"].tolist()
            marker_buffer.append((x, y, c, states))
            # alternate the side so loops don't collide across tasks
            # loop_buffer.append((x, y, c, 1 if (i % 2) else -1))
            loop_buffer.append((x, y, c, 1))

        # draw markers on top
        
        RING_STATES = {"Navigate to the Goal", "Idle"}
        BASE_SIZE = 64
        RING_SCALE = 1.85
        for (x, y, c, states) in marker_buffer:
            states = np.asarray(states)

            # boolean mask for ring points
            ring_mask = np.isin(states, list(RING_STATES))

            # 1) draw rings first (slightly larger, hollow)
            if ring_mask.any():
                ax.scatter(
                    x[ring_mask], y[ring_mask],
                    s=BASE_SIZE * (RING_SCALE**2),   # scale area, not radius
                    facecolors="none",
                    edgecolors=c,
                    linewidths=2.2,
                    zorder=9
                )

            # 2) draw filled centers for ALL points
            ax.scatter(
                x, y,
                s=BASE_SIZE,
                facecolors=c,
                edgecolors=c,
                linewidths=1.2,
                zorder=10
            )

        # draw loop symbols on top of markers
        for (x, y, c, side) in loop_buffer:
            draw_stationary_loops(ax, x, y, color=c, side=side, size=15, outline=outline)
        
        # Figure-level legend at the top (horizontal)
        handles, labels = ax.get_legend_handles_labels()
        ncols = min(len(handles), 7)  # spread across up to 7 columns
        fig.legend(
            handles, labels,
            loc="upper center", bbox_to_anchor=(0.5, 1.13),
            ncol=ncols, frameon=True, title="Task",
            title_fontsize=18, fontsize=18,
            handlelength=1.5, handletextpad=0.8, borderaxespad=0.5
        )

        # cosmetics
        n_states = len(subtask_map)
        ax.set_yticks(range(n_states))
        ax.set_yticklabels([subtask_map.get(i, "") for i in range(n_states)], fontsize=18)
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        #ax.legend(ncol=3, loc="upper left", fontsize=18, frameon=True, title="Task", title_fontsize=18)
        ax.set_xticks(x_int)
        ax.set_xticklabels(x_labels, fontsize=18, rotation=20, ha="right")
        ax.set_xlim(x_int.min() - 0.4, x_int.max() + 0.4)
        ax.set_ylim(-0.5, n_states - 0.5)

        if add_sentences:
            cols = [task_ids[:3], task_ids[3:]]
            x_cols = [-0.2, 0.45]
            line_spacing = 0.30
            y0 = 1.3
            for c, col_ids in enumerate(cols):
                y_row = y0
                for tid in col_ids:
                    text = f"T{tid}: {task_map.get(tid, '')}"
                    ax_desc.text(
                        x_cols[c], y_row, wrapper.fill(text),
                        ha="left", va="top", fontsize=18,
                        transform=ax_desc.transAxes, color=task_color(tid)
                    )
                    y_row -= line_spacing

        out_png = os.path.join(out_dir, f"progression_subtasks_{seq_id}_onehot.png")
        out_pdf = os.path.join(out_dir, f"progression_subtasks_{seq_id}_onehot.pdf")
        plt.savefig(out_png, dpi=200, bbox_inches="tight", pad_inches=0.2)
        plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.2)
        plt.close(fig)


def main():
    df = run_and_save()
    plot_task_accuracy(df)
    plot_subtask_timelines(df)
    plot_subtask_progression(df)
    print(f"Plots saved to: {PLOT_DIR}")

if __name__ == "__main__":
    main()
