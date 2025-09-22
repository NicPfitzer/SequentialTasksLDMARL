import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sequence_models.four_flags.model_training.rnn_model import EventRNN, NUM_AUTOMATA, MAX_SEQ_LEN, EVENT_DIM, STATE_DIM
from sequence_models.four_flags.model_training.mlp_decoder import Decoder
from sequence_models.four_flags.random_automaton_walk import R, G, B, P, SW
GL = 5
# task_map = {
#     0: "Locate the red flag, then the switch, and advance to the goal.",
#     1: "Search for the green flag, then the switch, and navigate to the goal.",
#     2: "Find the blue flag, spot the switch, and head for the target.",
#     3: "Identify the purple flag, navigate to the switch and proceed to the goal."
# }
task_map = {
    0:   "Locate the red flag, then the green one, and advance to the goal.",
    1:   "Find the red flag, spot the blue flag, and head for the target.",
    2:   "Identify the red flag, then the purple flag, and proceed to the goal.",

    3:   "Search for the green flag, then the red flag, and navigate to the goal.",
    4:   "Locate the green flag, then the blue flag, and move toward the target.",
    5:   "Find the green flag, identify the purple one, then reach the goal.",

    6:   "Spot the blue flag, then the red flag, and continue to the goal.",
    7:   "Head for the blue flag, then the green flag, and finish at the target.",
    8:   "Identify the blue flag, then the purple flag, and advance to the goal.",

    9:   "Locate the purple flag, then the red flag, and proceed to the goal.",
    10:  "Find the purple flag, then the green one, and head toward the target.",
    11:  "Search for the purple flag, then the blue flag, and reach the goal.",
}

subtask_map = {
    R: "Find Red Flag",
    G: "Find Green Flag",
    B: "Find Blue Flag",
    P: "Find Purple Flag",
    SW: "Navigate to the Switch",
    GL: "Navigate to the Goal",
}

print(subtask_map)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model from .pth checkpoint
def load_event_rnn_model(checkpoint_path: str = "event_rnn_best.pth", decoder: Decoder = None) -> EventRNN:
    if decoder is None:
        model = EventRNN(
            event_dim=5, y_dim=1024, latent_dim=1024, state_dim=10, input_dim=64, num_layers=1
        )
    else:
        print("Using provided decoder model.")
        model = EventRNN(
            event_dim=5, y_dim=1024, latent_dim=1024, state_dim=8, input_dim=64, num_layers=1, decoder=decoder
        )
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_decoder_model(path: str, emb_size: int = 1024, out_size: int = 6) -> Decoder:
    model = Decoder(emb_size, out_size)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


# Predict full sequence of state labels
def predict_state_rollout(task: str, subtask: str, events: list[list[float]], model: EventRNN) -> dict:
    # Embed sentence
    embedder = SentenceTransformer("thenlper/gte-large")
    y = embedder.encode(task, convert_to_tensor=True).unsqueeze(0)  # (1, y_dim)
    if subtask:
        h = embedder.encode(subtask, convert_to_tensor=True).unsqueeze(0)  # (1, y_dim)
    else:
        h = torch.zeros_like(y)

    T = len(events)
    e = torch.tensor(events, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, T, e_dim)
    y = y.unsqueeze(1).expand(-1, T, -1).to(DEVICE)                         # (1, T, y_dim)
    tasks = []

    with torch.no_grad():
        state_logits, _ = model._initalized_rollout(e, y, h)  # (1, T, state_dim)
        probs = torch.sigmoid(state_logits).squeeze(0)   # (T, state_dim)
        preds = (probs > 0.5).int().tolist()             # list of length T, each is list[state_dim]
        
        # Task
        # Automaton state is a binary vector of length NUM_AUTOMATA
        binarized = probs[:,:NUM_AUTOMATA]
        weights = 2 ** torch.arange(NUM_AUTOMATA - 1, -1, -1, device=binarized.device)
        task_index = (binarized * weights).sum(dim=1).int()
        tasks = [task_map[i.item()] for i in task_index]
        
        # Subtask
        subtask_index = probs[:,NUM_AUTOMATA:].argmax(dim=-1)  #
        print(f"Subtask index: {subtask_index}")
        subtasks = [subtask_map[i.item()] for i in subtask_index]


    return {
        "pred_probs": probs.cpu().tolist(),
        "pred_labels": preds,
        "tasks": tasks,
        "subtasks": subtasks,
    }

# Run
#decoder = load_decoder_model("sequence_models/four_flags/four_flags_decoder.pth")
model = load_event_rnn_model("sequence_models/four_flags/two_colors_rnn.pth")

# ]
events = [
    [0., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1.],
    [0., 1., 1., 0., 0.],
    [0., 0., 1., 0., 1.],
    [0., 0., 1., 1., 1.],
    # [1.,1.,1.,0.,0.],
    # [1.,1.,1.,1.,0.],
    # [1.,1.,1.,1.,1.],
]

task = "Find the purple flag, then the green one, and head toward the target."
subtask = None # Initialization subtask - We don't need to start at the beginning of the automaton.

result = predict_state_rollout(
    task=task,
    subtask=subtask,
    events=events,
    model=model
)

def pretty_print(task, subtask, events, result):
    # ----- Inputs -----
    print("---Input---")
    print(f"  Task    : {task.strip()}")
    if subtask:
        print(f"  Subtask : {subtask.strip()}")
    else:
        print("  Subtask : None")
    print("  Events  (Red,Green,Blue,Purple,Switch):")
    for idx, event in enumerate(events, 1):
        print(f"    {idx:>2}. {event}")

    # ----- Outputs -----
    print("\n---Outputs---")
    for t in range(len(result["pred_labels"])):
        label = result["pred_labels"][t]
        prob = result["pred_probs"][t]
        tk = result["tasks"][t]
        subtask = result["subtasks"][t]
        rounded_prob = np.round(prob, 3)
        print(f"  {t:>2}. Prob    : {rounded_prob}")
        print(f"      Label   : {label}")
        print(f"      Task    : {tk}")
        print(f"      Subtask : {subtask}")

pretty_print(task, subtask, events, result)

