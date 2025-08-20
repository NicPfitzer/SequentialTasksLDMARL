import random
import torch
import numpy as np
import json

from sequence_models.four_flags.model_training.rnn_model import EventRNN

import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Any, Dict, List, Optional

DECODER_OUTPUT_SIZE = 100
MAX_SEQ_LEN = 15
EVENT_DIM = 5

FIND_RED = 0
FIND_GREEN = 1
FIND_BLUE = 2
FIND_PURPLE = 3
FIND_SWITCH = 4
FIND_GOAL = 5

STATES = {
    "FIND_GOAL": FIND_GOAL,
    "goal": FIND_GOAL,
    "FIND_SWITCH": FIND_SWITCH,
    "switch": FIND_SWITCH,
    "FIND_RED": FIND_RED,
    "red": FIND_RED,
    "FIND_GREEN": FIND_GREEN,
    "green": FIND_GREEN,
    "FIND_BLUE": FIND_BLUE,
    "blue": FIND_BLUE,
    "FIND_PURPLE": FIND_PURPLE,
    "purple": FIND_PURPLE,
}

TextKeys = ("response", "text", "sentence", "prompt")
EmbedKeys = ("embedding", "vector")

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
    decoder_model = Decoder(emb_size= embedding_size, out_size=EVENT_DIM+1)
    decoder_model.load_state_dict(torch.load(model_path, map_location=device))
    decoder_model.eval()
    
def load_sequence_model(model_path, embedding_size, event_size, state_size, device):
    
    global sequence_model
    sequence_model = EventRNN(event_dim=event_size, y_dim=embedding_size, latent_dim=embedding_size, input_dim=64, state_dim=state_size, decoder=None)
    sequence_model.load_state_dict(torch.load(model_path, map_location=device))
    sequence_model.eval()

from pathlib import Path
import json
import torch
from typing import Dict, Any, Optional, List

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
        
    def load_multitask_data(self, json_path, device='cpu'):

        project_root = Path(__file__).resolve().parents[2]  # adjust if needed
        full_path = project_root / json_path

        with full_path.open('r') as f:
            data = json.load(f)

        def _get_first(d: Dict[str, Any], keys: tuple[str, ...]) -> Optional[Any]:
            for k in keys:
                if k in d:
                    return d[k]
            return None

        def process_dataset(data) -> Dict[str, Any]:
            """Return dict-of-lists with one big embedding tensor [N, D]."""

            states: List[str] = []
            responses: List[str] = []
            embs: List[List[float]] = []

            for state, records in data.items():
                if not isinstance(records, list):
                    raise ValueError(f"State '{state}' must map to a list, got {type(records)}")

                for rec in records:
                    if not isinstance(rec, dict):
                        rec = {"response": rec}

                    response = _get_first(rec, TextKeys)
                    embedding = _get_first(rec, EmbedKeys)

                    if not isinstance(response, str) or not response.strip():
                        continue

                    states.append(state)
                    responses.append(response.strip())
                    embs.append(embedding)

            embed_tensor = torch.tensor(embs, dtype=torch.float32, device=device)

            return {
                "state": states,
                "response": responses,
                "embedding": embed_tensor,   # shape [N, D]
            }

        self.train_dict = process_dataset(data)
        self.total_dict_size = len(self.train_dict["response"])
        
    def load_sequence_data(
        self,
        json_path,
        device='cpu'):

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

        self.train_dict = process_dataset(data)
        self.total_dict_size = len(next(iter(self.train_dict.values())))

    def sample_multitask_dataset(self, env_index: torch.Tensor, forced_state=None):
        
        if env_index is None:
            env_index = torch.arange(self.batch_size, device=self.device)
        else:
            env_index = torch.atleast_1d(torch.tensor(env_index, device=self.device))
        
        packet_size = env_index.shape[0]
        
        # --- pick indices ------------------------------------------------------------     # or any key with same length
        if packet_size <= self.total_dict_size:
            # Normal case: sample *without* replacement
            sample_indices = torch.randperm(self.total_dict_size, device=self.device)[:packet_size]
        else:
            # Need repeats → build “base” + “extra” indices
            repeats, remainder = divmod(packet_size, self.total_dict_size)

            # 1) repeat every index the same number of times
            base = torch.arange(self.total_dict_size, device=self.device).repeat(repeats)

            # 2) top-up with a random subset for the leftover slots
            extra = torch.randperm(self.total_dict_size, device=self.device)[:remainder] \
                    if remainder > 0 else torch.empty(0, dtype=torch.long, device=self.device)

            sample_indices = torch.cat([base, extra])
        
        # Sample tensor
        task_dict = {key: value[sample_indices] for key, value in self.train_dict.items() if key in self.train_dict and key not in ["response", "state"]}
        task_dict["response"] = [self.train_dict["response"][i] for i in sample_indices]
        task_dict["state"] = [self.train_dict["state"][i] for i in sample_indices]
        # Sample sentences
        indices_list = sample_indices.tolist()
        if "response" in self.train_dict:
            for i , idx in enumerate(env_index):
                self.response[idx] = task_dict["response"][i]

        if "embedding" in task_dict:
            self.subtask_embeddings[env_index] = task_dict["embedding"]

        if "state" in task_dict:
            for i , idx in enumerate(env_index):
                state = task_dict["state"][i]
                self.states[idx] = STATES[state] if state in STATES else 0
                
    def sample_sequence_dataset(self, env_index: torch.Tensor, forced_state=None):

        if env_index is None:
            env_index = torch.arange(self.batch_size, device=self.device)
        else:
            env_index = torch.atleast_1d(torch.tensor(env_index, device=self.device))

        packet_size = env_index.shape[0]

        # ---------- Build an inverted index: state_value -> [(sample_j, [matching_positions...]), ...]
        if "states" not in self.train_dict:
            raise ValueError("train_dict must contain 'states' sequences.")

        # values we can target
        all_state_values = list(STATES.values())
        # map: state_value -> list of (j, match_positions)
        candidates_by_state = {sv: [] for sv in all_state_values}

        for j in range(self.total_dict_size):
            seq_states = self.train_dict["states"][j]
            # collect match positions per state value
            buckets = {sv: [] for sv in all_state_values}
            for k in range(len(seq_states)):
                sv = STATES[seq_states[k]]  # map token/label -> canonical value
                if sv in buckets:
                    buckets[sv].append(k)
            for sv, pos in buckets.items():
                if pos:
                    candidates_by_state[sv].append((j, pos))

        # Fast lookup: which states actually exist
        available_states = [sv for sv, lst in candidates_by_state.items() if len(lst) > 0]

        # ---------- Choose per-slot target state, sample, and subtask index
        chosen_indices = []          # per-slot chosen sample j
        chosen_subtask_idxs = []     # per-slot chosen subtask position

        for _ in range(packet_size):
            # 1) pick target state (forced or random), but ensure it's available if possible
            target_sv = forced_state if forced_state is not None else random.choice(all_state_values)
            if target_sv not in available_states:
                if available_states:
                    target_sv = random.choice(available_states)
                else:
                    # no samples contain any known state; hard fallback
                    target_sv = None

            # 2) pick random sample with that state
            if target_sv is not None and candidates_by_state[target_sv]:
                j, positions = random.choice(candidates_by_state[target_sv])
                sub_idx = random.choice(positions)
            else:
                # global fallback: arbitrary sample/subtask 0
                j = random.randrange(self.total_dict_size)
                sub_idx = 0

            chosen_indices.append(j)
            chosen_subtask_idxs.append(sub_idx)

        chosen_indices_t = torch.tensor(chosen_indices, device=self.device, dtype=torch.long)

        # ---------- Assemble task_dict consistently for the chosen samples
        task_dict = {}

        # tensor-like entries: gather by index
        tensor_keys = [k for k, v in self.train_dict.items()
                    if k not in ["states", "subtasks", "responses", "summary"] and torch.is_tensor(v)]
        for key in tensor_keys:
            task_dict[key] = self.train_dict[key][chosen_indices_t]

        # ragged/list-like entries: pick one-by-one
        for key in ["summary", "responses", "states", "subtasks"]:
            if key in self.train_dict:
                task_dict[key] = [self.train_dict[key][j] for j in chosen_indices]

        # ---------- Write fixed fields that don't depend on subtask position
        if "task" in task_dict:
            self.task_embeddings[env_index] = task_dict["task"]

        if "summary" in task_dict:
            for i, idx in enumerate(env_index):
                self.summary[idx] = task_dict["summary"][i]

        if "event" in task_dict:
            self.event_sequence[env_index] = task_dict["event"]

        # ---------- Per-slot: commit chosen subtask position, response, and sequence_length
        if "subtasks" in task_dict and "responses" in task_dict and "states" in task_dict:
            for i, idx in enumerate(env_index):
                num_subtasks = task_dict["subtasks"][i].shape[0] if hasattr(task_dict["subtasks"][i], "shape") \
                            else len(task_dict["subtasks"][i])
                sub_idx = chosen_subtask_idxs[i] if num_subtasks > 0 else 0
                sub_idx = min(max(sub_idx, 0), max(num_subtasks - 1, 0))

                self.sequence_length[idx] = sub_idx
                self.response[idx] = task_dict["responses"][i][sub_idx]

                # normalization from your original code
                if self.sequence_length[idx] == 0:
                    self.event_sequence[idx].zero_()
                    self.sequence_length[idx] = 1
        else:
            # minimal fallback if ragged keys are missing
            for i, idx in enumerate(env_index):
                self.sequence_length[idx] = 1
                if "responses" in task_dict and len(task_dict["responses"][i]) > 0:
                    self.response[idx] = task_dict["responses"][i][0]
                if "event" in task_dict:
                    self.event_sequence[idx] = task_dict["event"][i]

        # ---------- Rollout embeddings after everything is aligned
        if "subtasks" in task_dict:
            self.compute_subtask_embedding_rollout_from_rnn(env_index)

        
    # def sample_sequence_dataset(self, env_index: torch.Tensor, forced_state=None):
        
    #     if env_index is None:
    #         env_index = torch.arange(self.batch_size, device=self.device)
    #     else:
    #         env_index = torch.atleast_1d(torch.tensor(env_index, device=self.device))
        
    #     packet_size = env_index.shape[0]
        
    #     # --- pick indices ------------------------------------------------------------     # or any key with same length
    #     if packet_size <= self.total_dict_size:
    #         # Normal case: sample *without* replacement
    #         sample_indices = torch.randperm(self.total_dict_size, device=self.device)[:packet_size]
    #     else:
    #         # Need repeats → build “base” + “extra” indices
    #         repeats, remainder = divmod(packet_size, self.total_dict_size)

    #         # 1) repeat every index the same number of times
    #         base = torch.arange(self.total_dict_size, device=self.device).repeat(repeats)

    #         # 2) top-up with a random subset for the leftover slots
    #         extra = torch.randperm(self.total_dict_size, device=self.device)[:remainder] \
    #                 if remainder > 0 else torch.empty(0, dtype=torch.long, device=self.device)

    #         sample_indices = torch.cat([base, extra])
        
    #     # Sample tensors
    #     task_dict = {key: value[sample_indices] for key, value in self.train_dict.items() if key in self.train_dict and key not in ["states", "subtasks", "responses", "summary"]}
    #     # Sample sentences
    #     indices_list = sample_indices.tolist()
    #     if "summary" in self.train_dict:
    #         task_dict["summary"] = [self.train_dict["summary"][i] for i in indices_list]

    #     if "responses" in self.train_dict:
    #         task_dict["responses"] = [self.train_dict["responses"][i] for i in indices_list]

    #     if "states" in self.train_dict:
    #         task_dict["states"] = [self.train_dict["states"][i] for i in indices_list]

    #     if "subtasks" in self.train_dict:
    #         task_dict["subtasks"] = [self.train_dict["subtasks"][i] for i in indices_list]

    #     if "task" in task_dict:
    #         self.task_embeddings[env_index] = task_dict["task"]
        
    #     if "summary" in task_dict:
    #         for i , idx in enumerate(env_index):
    #             self.summary[idx] = task_dict["summary"][i]
        
    #     if "event" in task_dict:
    #         event = task_dict["event"]
    #         self.event_sequence[env_index] = event
        
    #     if "subtasks" in task_dict and "responses" in task_dict and "states" in task_dict:
    #         for i , idx in enumerate(env_index):
    #             num_subtasks = task_dict["subtasks"][i].shape[0]
    #             states = task_dict["states"][i]
    #             # Chose random subtask index with equal probability to land on each subtask
    #             state_pick = random.choice(list(STATES.values()))
    #             subtask_idx =  random.randint(0, num_subtasks - 1) if num_subtasks > 0 else 0
    #             attempt = 0
    #             while STATES[states[subtask_idx]] != state_pick and attempt < 10:
    #                 subtask_idx = random.randint(0, num_subtasks - 1) if num_subtasks > 0 else 0
    #                 attempt += 1
    #             self.sequence_length[idx] = subtask_idx
    #             self.response[idx] = task_dict["responses"][i][subtask_idx]
                
    #             if self.sequence_length[idx] == 0:
    #                 self.event_sequence[idx].zero_()
    #                 self.sequence_length[idx] = 1
                
    #     if "subtasks" in task_dict:
    #         self.compute_subtask_embedding_rollout_from_rnn(env_index)   
    
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
        
            
