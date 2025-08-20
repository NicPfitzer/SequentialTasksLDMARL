import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from sentence_transformers import SentenceTransformer
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from sequence_models.four_flags.model_training.rnn_model import EventRNN
from sequence_models.four_flags.model_training.mlp_decoder import Decoder
from sequence_models.four_flags.random_automaton_walk import R, G, B, P, SW
GL = 5
task_map = {
    0: "Find flags in rgbp order, hit the switch and navigate to the goal",
}

subtask_map = {
    SW: "Navigate to the Switch",
    GL: "Navigate to the Goal",
    R: "Find Red Flag",
    G: "Find Green Flag",
    B: "Find Blue Flag",
    P: "Find Purple Flag",
}

print(subtask_map)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model from .pth checkpoint
def load_event_rnn_model(checkpoint_path: str = "event_rnn_best.pth", decoder: Decoder = None) -> EventRNN:
    if decoder is None:
        model = EventRNN(
            event_dim=5, y_dim=1024, latent_dim=1024, state_dim=7, input_dim=64, num_layers=1
        )
    else:
        print("Using provided decoder model.")
        model = EventRNN(
            event_dim=5, y_dim=1024, latent_dim=1024, state_dim=7, input_dim=64, num_layers=1, decoder=decoder
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
        task_index = probs[:,:1].argmax(dim=-1)  # (T,)
        tasks = [task_map[i.item()] for i in task_index]
        
        # Subtask
        subtask_index = probs[:,1:].argmax(dim=-1)  #
        print(f"Subtask index: {subtask_index}")
        subtasks = [subtask_map[i.item()] for i in subtask_index]


    return {
        "pred_probs": probs.cpu().tolist(),
        "pred_labels": preds,
        "tasks": tasks,
        "subtasks": subtasks,
    }

# Run
decoder = load_decoder_model("sequence_models/four_flags/four_flags_decoder.pth")
model = load_event_rnn_model("sequence_models/four_flags/event_rnn_best_gru-in64-bs128.pth", decoder)

# ]
events = [
    [0.,0.,0.,0.,0.],
    [1.,0.,0.,0.,0.],
    [1.,1.,0.,0.,0.],
    [1.,1.,1.,0.,0.],
    [1.,1.,1.,1.,0.],
    [1.,1.,1.,1.,1.],
]

task = "Hunt flags, flip the switch, and race to the finish!\n"
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

