"""Event RNN (v2) – hidden == predicted h
===========================================================
This revision unifies the GRU hidden state and the target vector **hₜ** so
`prev_h` *is literally the GRU hidden*.  Concretely, the GRU hidden size
is now **h_dim = 1024**, and the projection layer `next_state_head` has
been dropped.

Public interface (constructor arguments, loss breakdown, training loop)
remains the same, but the internal tensor shapes are simpler:

```
prev_h ∈ ℝ^{B×h_dim}
gru_hidden ≡ prev_h
```

So the autoregressive update is now just

```python
gru_hidden = rnn_cell(fused_t, gru_hidden)  # new hidden == new hₜ
next_h     = gru_hidden
```

The rest of the file is a complete, runnable script.
"""

from __future__ import annotations

import json
import math
import pathlib
from typing import Dict, List, Tuple

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split
import itertools

import wandb
from pytorch_lightning.loggers import WandbLogger


###############################################################################
# Constants – unchanged
###############################################################################

MAX_SEQ_LEN = 15
EVENT_DIM = 5
STATE_DIM = 8
NUM_AUTOMATA = STATE_DIM - EVENT_DIM - 1

###############################################################################
# Dataset + DataLoader helpers – identical to v1 (see comments there)
###############################################################################

def make_loaders(
    json_path: str,
    batch_size: int = 128,
    train_frac: float = 0.8,
    *,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    full_ds = EventSequenceDataset.from_json(json_path)
    n_total = len(full_ds)
    n_train = math.floor(n_total * train_frac)
    n_val = n_total - n_train

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full_ds, (n_train, n_val), generator=gen)

    kwargs = dict(
        batch_size=batch_size,
        collate_fn=EventSequenceDataset.collate,
        num_workers=0,
        pin_memory=True,
    )
    return (
        DataLoader(train_ds, shuffle=True, **kwargs),
        DataLoader(val_ds, shuffle=True, **kwargs),
    )


class EventSequenceDataset(Dataset):
    """Parses raw JSON into tensor dicts."""

    def __init__(self, raw: List[Dict]):
        self.samples: List[Dict[str, Tensor]] = []
        for ix, s in enumerate(raw):
            e_key = "events" if "events" in s else "e"
            if e_key not in s or not all(k in s for k in ("y", "h", "label","subtask_decoder_labels")):
                raise KeyError(f"Sample {ix} missing required keys")

            e = torch.tensor(s[e_key], dtype=torch.float32)
            h = torch.tensor(s["h"], dtype=torch.float32)
            y = torch.tensor(s["y"], dtype=torch.float32)
            subtask_label = torch.tensor(s["subtask_decoder_labels"], dtype=torch.float32)
            state_label = torch.tensor(s["label"], dtype=torch.float32)
            self.samples.append({"e": e, "y": y, "h": h, "label": state_label, "subtask_label": subtask_label})

    @classmethod
    def from_json(cls, path: str | pathlib.Path) -> "EventSequenceDataset":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
            if not isinstance(raw, list):
                raise ValueError("JSON must be an array of samples")
        return cls(raw)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]

    @staticmethod
    def collate(batch: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        lengths = torch.tensor([s["e"].size(0) for s in batch])
        max_len = MAX_SEQ_LEN
        event_dim = batch[0]["e"].size(1)
        h_dim = batch[0]["h"].size(1)
        label_dim = batch[0]["label"].size(1)
        subtask_dim = batch[0]["subtask_label"].size(1)

        e_out, h_out, y_out, labels, subtask_labels = [], [], [], [], []
        for s in batch:
            T = s["e"].size(0)
            pad_e = torch.zeros((max_len - T, event_dim))
            pad_h = torch.zeros((max_len - T, h_dim))
            pad_label = torch.zeros((max_len - T,label_dim))
            pad_subtask = torch.zeros((max_len - T, subtask_dim))

            e_out.append(torch.cat([s["e"], pad_e]))
            h_out.append(torch.cat([s["h"], pad_h]))
            y_rep = s["y"].unsqueeze(0).expand(max_len, -1)
            y_out.append(y_rep)
            labels.append(torch.cat([s["label"], pad_label]))
            subtask_labels.append(torch.cat([s["subtask_label"], pad_subtask]))

        return {
            "e": torch.stack(e_out),
            "h": torch.stack(h_out),
            "y": torch.stack(y_out),
            "subtask_labels": torch.stack(subtask_labels),
            "label": torch.stack(labels),
            "lengths": lengths
        }

###############################################################################
# EventRNN – hidden == h
###############################################################################

class EventRNN(pl.LightningModule):
    """RNN variant where the GRU hidden state *is* the predicted hₜ."""

    def __init__(
        self,
        *,
        event_dim: int,
        y_dim: int,
        latent_dim: int,
        state_dim: int,
        input_dim: int = 16,
        num_layers: int = 1,
        lr: float = 1e-4,
        cls_loss_weight: float = 2.0,
        ground_truth_h_rate: float = 0.25,
        recon_loss: str = "cosine",
        decoder: Decoder = None,
        use_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        D = latent_dim  # hidden size now equals target size
        I = input_dim
        
        if decoder is None:
            self.use_label_decoder = False
        else:
            self.use_label_decoder = True


        # ─── Embeddings ──────────────────────────────────────────────
        self.e_proj = nn.Linear(event_dim, I)
        hidden_dim = max(D // 2, 4)
        self.y_proj = nn.Sequential(
            nn.Linear(y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, I),
        )
        # self.h_proj = nn.Identity()  # no change in dimensionality
        # self.fuse = nn.Linear(2 * I, I)

        # ─── GRU stack ───────────────────────────────────────────────
        if num_layers == 1:
            self.rnn = nn.GRUCell(2*I, D)
        else:
            self.rnn = nn.GRU(2*I, D, num_layers=num_layers, batch_first=True)
            # we will still step manually to keep parity with TBPTT logic

        # subtask embedding Encoder
        self.use_encoder = use_encoder
        if self.use_encoder:
            self.subtask_encoder = nn.Sequential(
                nn.Linear(D, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, D),
            )

        # ─── Classification head ────────────────────────────────────
        hidden_dim = max(D // 2, 4)
        self.state_head = nn.Sequential(
            nn.Linear(D + 2*I, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

        self.recon_head = nn.Identity()  # no reconstruction head
        self.recon_decoder = decoder

        self.recon_loss = recon_loss.lower()
        if self.recon_loss not in {"mse", "cosine"}:
            raise ValueError("recon_loss must be 'mse' or 'cosine'")
        self.cls_w = cls_loss_weight
        self.lr = lr
        self.ground_truth_h_rate = ground_truth_h_rate

    # ------------------------------------------------------------------
    def _rollout(self, e: Tensor, y: Tensor, lengths: Tensor) -> Tensor:
        B, T, _ = e.shape
        D = self.hparams.latent_dim

        fused = torch.cat([self.e_proj(e),self.y_proj(y)], dim=-1)  # (B, T, 2*I)

        # ───── Manual stepping for GRUCell ─────
        if isinstance(self.rnn, nn.GRUCell):
            h = e.new_zeros(B, D)
            preds = []
            for t in range(T):
                h = self.rnn(fused[:, t], h)
                preds.append(h.clone())
            out = torch.stack(preds, dim=1)  # (B, T, D)
        else:                                                    # multi-layer path
            # ↓ Optional but recommended: pack to skip padding tokens
            packed = nn.utils.rnn.pack_padded_sequence(
                fused, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            h0 = fused.new_zeros(self.hparams.num_layers, B, self.hparams.latent_dim)
            packed_out, _ = self.rnn(packed, h0)                 # (ΣT,D)
            out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )                                # (B,T,D)

        state_decoder_out = self.state_head(torch.cat([out,fused], dim=-1))  # (B, T, state_dim)
        recon_out = self.recon_head(out)  # (B, T, state_dim)

        return state_decoder_out, recon_out
    
    def train_rollout(self, e: Tensor, y: Tensor, h_gt: Tensor) -> Tensor:
        
        B, T, _ = e.shape
        D = self.hparams.latent_dim
        fused = torch.cat([self.e_proj(e),self.y_proj(y)], dim=-1)  # (B, T, 2*I)

        # ───── Manual stepping for GRUCell ─────
        h = e.new_zeros(B, D)
        preds = []
        for t in range(T):
            
            if self.use_encoder and h_gt is not None and torch.rand(1).item() < self.ground_truth_h_rate:
                h = self.subtask_encoder(h_gt[:, t])
                
            h = self.rnn(fused[:, t], h)
            # Randomily sometimes replace the latest hidden state with the ground truth
            preds.append(h.clone())
            
        out = torch.stack(preds, dim=1)  # (B, T, D)                                              # (B,T,D)

        state_decoder_out = self.state_head(torch.cat([out,fused], dim=-1)) # (B, T, state_dim)
        recon_out = self.recon_head(out)  # (B, T, state_dim)
        if self.use_label_decoder:
            decoder_label = self.recon_decoder(recon_out)  # (B, T, LABEL_SIZE)
        else:
            decoder_label = None

        return state_decoder_out, recon_out, decoder_label

    def _initalized_rollout(self, e: Tensor, y: Tensor, h: Tensor) -> Tensor:
        B, T, _ = e.shape
        D = self.hparams.latent_dim
        L = self.hparams.num_layers

        assert h.size(0) == B and h.size(1) == D, "h must be of shape (B, D)"
        
        fused = torch.cat([self.e_proj(e),self.y_proj(y)], dim=-1)  # (B, T, 2*I)

        # ───── Manual stepping for GRUCell ─────
        if isinstance(self.rnn, nn.GRUCell):
            preds = []
            for t in range(T):
                h = self.rnn(fused[:, t], h)
                preds.append(h.clone())
            out = torch.stack(preds, dim=1)  # (B, T, D)

        state_decoder_out = self.state_head(torch.cat([out,fused], dim=-1))  # (B, T, state_dim)
        recon_out = self.recon_head(out)  # (B, T, state_dim)

        return state_decoder_out, recon_out

    def _forward(self, e: Tensor, y: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        
        """Forward step of the RNN"""
        fused = torch.cat([self.e_proj(e),self.y_proj(y)], dim=-1)  # (B, T, 2*I)
        h = self.rnn(fused, h)
        state_decoder_out = self.state_head(torch.cat([h,fused],dim=-1))  # (B, T, state_dim)

        return h, state_decoder_out

    def convert_to_latent(self, g: Tensor) -> Tensor:
        return self.subtask_encoder(g)

    # ------------------------------------------------------------------
    def _step(self, batch):
        e, y, target_h = batch["e"], batch["y"], batch["h"] # (B, T, e_dim), (B, T, y_dim), (B, T+1, D)
        lengths = batch["lengths"] # (B,)
        labels = batch["label"] # (B,T+1, state_dim)
        decoder_labels = batch["subtask_labels"] # (B,T+1, subtask_dim)
        
        #Only compute reconstruction loss for positive labels      
        target = target_h[:, 1:]  # (B, T, D)
        label = labels[:, 1:]  # (B, T, state_dim)
        decoder_label = decoder_labels[:, 1:]  # (B, T, subtask_dim)

        #pred_cls, pred_recon, pred_grid = self._rollout(e, y, lengths)
        pred_cls, pred_recon, pred_decoder_label = self.train_rollout(e, y, target_h)  # (B, T, state_dim), (B, T, D)

        # Mask: (B, T)
        mask = (
            torch.arange(target.size(1), device=lengths.device)
            .unsqueeze(0).expand(lengths.size(0), -1)
            < lengths.unsqueeze(1)
        )

        #================ Reconstruction Loss Calculation ==========
        if self.recon_loss == "mse":
            mse = F.mse_loss(pred_recon, target, reduction='none').mean(dim=-1)
            loss_recon_h = (mse * mask).sum() / mask.sum()
        else:
            cos = F.cosine_similarity(pred_recon, target, dim=-1)  # (B, T)
            loss_recon_h = ((1 - cos) * mask).sum() / mask.sum() 

        # loss_grid = F.binary_cross_entropy_with_logits(pred_grid, grid, reduction='none').mean(dim=-1)  # (B, T)
        # # gt_grid = self.recon_decoder(target)  # (B, T, GRID_SIZE*GRID_SIZE + NUM_CLASSES)
        # # loss_grid_gt = F.binary_cross_entropy_with_logits(gt_grid, grid, reduction='none').mean(dim=-1)  # (B, T)
        # # masked_loss_grid = abs(loss_grid - loss_grid_gt) * mask
        # masked_loss_grid = loss_grid * mask  # (B, T)
        # loss_recon_grid = masked_loss_grid.sum() / (mask.sum())
        # loss_recon = loss_recon_h * 0 + loss_recon_grid

        #================ Classification Loss Calculation ==========
        raw_loss_cls = F.binary_cross_entropy_with_logits(pred_cls, label, reduction='none').mean(dim=-1) # (B, T, state_dim)
        masked_loss = raw_loss_cls * mask
        loss_cls = masked_loss.sum() / mask.sum()
        
        if self.use_label_decoder:
            #label_2 = torch.sigmoid(self.recon_decoder(target))  # (B, T, LABEL_SIZE)
            label_loss = F.binary_cross_entropy_with_logits(
                pred_decoder_label,decoder_label, reduction='none'
            ).mean(dim=-1)
            masked_losslabel_loss = label_loss * mask
            label_loss = masked_losslabel_loss.sum() / mask.sum()
        else:
            label_loss = 0

        loss = self.cls_w * loss_cls.clone() + label_loss

        return loss, loss_cls, loss_recon_h, label_loss


    # ----------------------------§--------------------------------------
    def training_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, loss_cls, loss_recon_h, label_loss = self._step(batch)
        self.log_dict({
            "train_loss": loss,
            "train_loss_cls": loss_cls,
            "train_loss_recon": loss_recon_h,
            "train_label_loss": label_loss,
        }, prog_bar=True)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int):
        loss, loss_cls, loss_recon_h, label_loss = self._step(batch)
        self.log_dict({
            "val_loss": loss,
            "val_loss_cls": loss_cls,
            "val_loss_recon": loss_recon_h,
            "val_label_loss": label_loss,
        }, prog_bar=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def run_training(batch_size: int, input_dim: int, resume_ckpt: str | None = None, decoder_path: str = "sequence_models/four_rooms/four_rooms_decoder.pth"):
    train_loader, val_loader = make_loaders(
        "sequence_models/data/dataset_two_color_ALL.json", batch_size=batch_size
    )
    
    # decoder = Decoder(
    #     emb_size=1024,
    #     out_size=EVENT_DIM+1,
    # )
    # #Initialize the model
    # decoder.load_state_dict(torch.load(decoder_path, map_location="mps"))
    # decoder.eval() # Freeze the decoder
    # for param in decoder.parameters():
    #     param.requires_grad = False

    model = EventRNN(
        lr=5e-5,
        event_dim=EVENT_DIM,
        y_dim=1024,
        latent_dim=1024,
        state_dim=STATE_DIM,
        input_dim=input_dim,
        num_layers=1,
        cls_loss_weight=8.0,
        ground_truth_h_rate=0.0,
        decoder=None,
    )

    run_name = f"gru-in{input_dim}-bs{batch_size}"

    checkpoint_cb = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename=f"{run_name}" + "-{epoch:02d}-{val_loss:.4f}",
    )

    wandb_logger = WandbLogger(
        project="event-rnn",
        name=run_name,
        log_model=False
    )

    trainer = pl.Trainer(
        max_epochs=2000,
        accelerator="auto",
        log_every_n_steps=10,
        callbacks=[checkpoint_cb],
        gradient_clip_val=1.0,
        logger=wandb_logger
    )
    
    # trainer = pl.Trainer(
    #     max_epochs=1000,
    #     accelerator="auto",
    #     log_every_n_steps=10,
    #     callbacks=[checkpoint_cb],
    #     gradient_clip_val=1.0,
    #     logger=False
    # )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume_ckpt  # This enables resuming from checkpoint
    )


    best_ckpt = checkpoint_cb.best_model_path
    best_model = EventRNN.load_from_checkpoint(best_ckpt)
    save_path = f"sequence_models/four_flags/event_rnn_best_{run_name}.pth"

    torch.save(best_model.state_dict(), save_path)
    print(f"[{run_name}] Best model saved to {save_path} (ckpt: {best_ckpt})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    batch_sizes = [128]
    input_dims = [64]

    for bs, inp in itertools.product(batch_sizes, input_dims):
        run_training(bs, inp, resume_ckpt=args.resume)
        wandb.finish()
