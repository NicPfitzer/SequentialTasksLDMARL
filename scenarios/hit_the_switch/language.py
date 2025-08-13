import random
import torch
import numpy as np
import json

from sequence_models.hit_the_switch.model_training.rnn_model import EventRNN

import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

DECODER_OUTPUT_SIZE = 100
MAX_SEQ_LEN = 4
EVENT_DIM = 1

FIND_SWITCH = 0
FIND_GOAL = 1
STATES = {
    "FIND_GOAL": FIND_GOAL,
    "G": FIND_GOAL,
    "FIND_SWITCH": FIND_SWITCH,
    "S": FIND_SWITCH
}

train_dict = None
total_dict_size = None
data_grid_size = None
decoder_model = None

class Decoder(nn.Module):
    def __init__(self, emb_size, out_size, hidden_size=128):
        super().__init__()
        self.l0 = nn.Linear(emb_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, out_size)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.act(self.l0(x))
        return torch.sigmoid(self.l1(x))

def load_decoder(model_path, embedding_size, device):
    
    global decoder_model
    decoder_model = Decoder(emb_size= embedding_size, out_size=DECODER_OUTPUT_SIZE+4)
    decoder_model.load_state_dict(torch.load(model_path, map_location=device))
    decoder_model.eval()
    
def load_sequence_model(model_path, embedding_size, event_size, state_size, device):
    
    global sequence_model
    sequence_model = EventRNN(event_dim=event_size, y_dim=embedding_size, latent_dim=embedding_size, input_dim=64, state_dim=state_size).to(device)
    sequence_model.load_state_dict(torch.load(model_path, map_location=device))
    sequence_model.eval()
    
def load_task_data(
    json_path,
    device='cpu'):
    global train_dict
    global total_dict_size

    # Resolve path to ensure it's absolute and correct regardless of cwd
    project_root = Path(__file__).resolve().parents[2]  # Adjust depending on depth of current file
    full_path = project_root / json_path

    with full_path.open('r') as f:
        data = json.load(f)

    if isinstance(data, list) and len(data) > 1 and all(isinstance(d, dict) for d in data):
        np.random.shuffle(data)
    else:
        data = [data]  # Ensure data is a list of dicts

    def process_dataset(dataset):
        output = {}

        if all("states" in entry for entry in dataset):
            states = [entry["states"] for entry in dataset]
            output["states"] = states
        
        if all("y" in entry for entry in dataset):
            task = [entry["y"] for entry in dataset]
            output["task"] = torch.tensor(task, dtype=torch.float32, device=device)
        
        if all("h" in entry for entry in dataset):
            embeddings = [torch.tensor(entry["h"],dtype=torch.float32, device=device) for entry in dataset]
            output["subtasks"] = embeddings
        
        if all("events" in entry for entry in dataset):
            events = []
            for entry in dataset:
                e_all = torch.zeros((MAX_SEQ_LEN, EVENT_DIM), dtype=torch.float32, device=device)
                e = torch.tensor(entry["events"], dtype=torch.float32, device=device)
                e_all[:e.shape[0], :] = e
                events.append(e_all)
            output["event"] = torch.stack(events)
        
        if all("summary" in entry for entry in dataset):
            sentences = [entry["summary"] for entry in dataset]
            output["summary"] = sentences
        
        if all("responses" in entry for entry in dataset):
            responses = [entry["responses"] for entry in dataset]
            output["responses"] = responses
        
        return output

    train_dict = process_dataset(data)
    total_dict_size = len(next(iter(train_dict.values())))
    

class LanguageUnit:

    def __init__(self, batch_size, embedding_size, use_embedding_ratio, device='cpu'):

        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.use_embedding_ratio = use_embedding_ratio
        self.device = device

        # Task
        self.embedding_size = embedding_size
        self.task_embeddings = torch.zeros((self.batch_size,embedding_size),device=self.device)
        self.subtask_embeddings = torch.zeros((self.batch_size,embedding_size),device=self.device)
        self.event_sequence = torch.zeros((self.batch_size,MAX_SEQ_LEN,EVENT_DIM),device=self.device)
        self.sequence_length = torch.zeros((self.batch_size,), dtype=torch.int, device=self.device)
        self.states = torch.zeros((self.batch_size,), dtype=torch.int64, device=self.device)
        self.summary = [ "" for _ in range(self.batch_size)]
        self.response = [ "" for _ in range(self.batch_size)]
        
    def sample_dataset(self, env_index: torch.Tensor, forced_state=None):
        
        if env_index is None:
            env_index = torch.arange(self.batch_size, device=self.device)
        else:
            env_index = torch.atleast_1d(torch.tensor(env_index, device=self.device))
        
        packet_size = env_index.shape[0]
        
        # --- pick indices ------------------------------------------------------------     # or any key with same length
        if packet_size <= total_dict_size:
            # Normal case: sample *without* replacement
            sample_indices = torch.randperm(total_dict_size, device=self.device)[:packet_size]
        else:
            # Need repeats → build “base” + “extra” indices
            repeats, remainder = divmod(packet_size, total_dict_size)

            # 1) repeat every index the same number of times
            base = torch.arange(total_dict_size, device=self.device).repeat(repeats)

            # 2) top-up with a random subset for the leftover slots
            extra = torch.randperm(total_dict_size, device=self.device)[:remainder] \
                    if remainder > 0 else torch.empty(0, dtype=torch.long, device=self.device)

            sample_indices = torch.cat([base, extra])
        
        # Sample tensors
        task_dict = {key: value[sample_indices] for key, value in train_dict.items() if key in train_dict and key not in ["states", "subtasks", "responses", "summary"]}
        # Sample sentences
        indices_list = sample_indices.tolist()
        if "summary" in train_dict:
            task_dict["summary"] = [train_dict["summary"][i] for i in indices_list]
            
        if "responses" in train_dict:
            task_dict["responses"] = [train_dict["responses"][i] for i in indices_list]
            
        if "states" in train_dict:
            task_dict["states"] = [train_dict["states"][i] for i in indices_list]
            
        if "subtasks" in train_dict:
            task_dict["subtasks"] = [train_dict["subtasks"][i] for i in indices_list]
        
        if "task" in task_dict:
            self.task_embeddings[env_index] = task_dict["task"]
        
        if "summary" in task_dict:
            for i , idx in enumerate(env_index):
                self.summary[idx] = task_dict["summary"][i]
        
        if "event" in task_dict:
            event = task_dict["event"]
            self.event_sequence[env_index] = event
        
        if "subtasks" in task_dict and "responses" in task_dict and "states" in task_dict:
            for i , idx in enumerate(env_index):
                num_subtasks = task_dict["subtasks"][i].shape[0]
                states = task_dict["states"][i]
                # Chose random subtask index with equal probability to land on each subtask
                rnd = random.random()
                state_pick = STATES["FIND_GOAL"] if rnd < 0.5 else STATES["FIND_SWITCH"]
                subtask_idx =  random.randint(0, num_subtasks - 1) if num_subtasks > 0 else 0
                attempt = 0
                while STATES[states[subtask_idx]] != state_pick and attempt < 10:
                    subtask_idx = random.randint(0, num_subtasks - 1) if num_subtasks > 0 else 0
                    attempt += 1
                self.sequence_length[idx] = subtask_idx
                self.response[idx] = task_dict["responses"][i][subtask_idx]
                
                if self.sequence_length[idx] == 0:
                    self.event_sequence[idx].zero_()
                    self.sequence_length[idx] = 1
                
        if "subtasks" in task_dict:
            self.compute_subtask_embedding_rollout_from_rnn(env_index)
    
    def get_subtask_embedding_from_rnn(self, env_index: torch.Tensor) -> torch.Tensor:
        """ Get the subtask embedding from the RNN model for the given enironments. """
        
        # Get the subtask embeddings for the given environments
        return self.subtask_embeddings[env_index].unsqueeze(1)
    
    def observe_task_embeddings(self):

        return self.task_embeddings.flatten(start_dim=1,end_dim=-1)
    
    def observe_subtask_embeddings(self):

        return self.subtask_embeddings.flatten(start_dim=1,end_dim=-1)

    def compute_subtask_embedding_rollout_from_rnn(self, env_index: torch.Tensor):
        """ Get the subtask embedding from the RNN model for the given environments. """
        e = self.event_sequence[env_index] # (B, MAX_SEQ_LEN, event_dim)
        y = self.task_embeddings[env_index].unsqueeze(1).expand(-1, MAX_SEQ_LEN, -1)  # (B, MAX_SEQ_LEN, emb_size)
        lengths = self.sequence_length[env_index]  # (B,)
        
        mask = (
            torch.arange(e.size(1), device=lengths.device)
            .unsqueeze(0).expand(lengths.size(0), -1)
            < lengths.unsqueeze(1)
        )
        
        state_one_hot_logits, sequence = sequence_model._rollout(e, y, lengths)
        state_one_hot = F.sigmoid(state_one_hot_logits) * mask.unsqueeze(-1)  # (B, MAX_SEQ_LEN, state_dim + autonmaton_dim)
        sequence = sequence * mask.unsqueeze(-1)  # (B, MAX_SEQ_LEN, emb_size) 
        # Decode the state one_hot into a state index
        # First two values are Automaton index. Next 4 values are state one-hot encoding
        states = torch.argmax(state_one_hot[:,:,1:],dim=-1)
        state_index = states[torch.arange(env_index.size(0)), lengths - 1]
        subtask = sequence[torch.arange(env_index.size(0)), lengths - 1, :]

        # Map rnn state representation to the environment
        state_index = state_index
        self.states[env_index] = state_index
        self.subtask_embeddings[env_index] = subtask
    
    def compute_forward_rnn(self, event: torch.Tensor, y: torch.Tensor, h: torch.Tensor):
        """
        Compute the next state of the RNN for the given environments.
        """
        next_h, state_decoder_out = sequence_model._forward(e=event, y=y, h=h)

        return next_h, state_decoder_out
        
        
    def reset_all(self):
        
        self.task_embeddings.zero_()
        self.subtask_embeddings.zero_()
        self.states.zero_()
        self.event_sequence.zero_()
        self.sequence_length.zero_()
        self.summary = [ ""  for _ in range(self.batch_size)]
        self.response = [ ""  for _ in range(self.batch_size)]
    
    def reset_env(self, env_index):

        self.task_embeddings[env_index].zero_()
        self.subtask_embeddings[env_index].zero_()
        self.states[env_index].zero_()
        self.event_sequence[env_index].zero_()
        self.sequence_length[env_index] = 0
        self.summary[env_index] = ""
        self.response[env_index] = ""
        
            
