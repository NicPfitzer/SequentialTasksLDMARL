#!/usr/bin/env python
"""
seq_emb_builder_multi_dir.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• Uses ONE shared dataset for ALL sequences.
• Input options:
    – List of explicit sequence files (SEQ_JSONS), or
    – A directory containing many *.json sequence files (SEQ_DIR).
• Writes one output per input. If SEQ_DIR is used, outputs go to that directory
  with auto names: dataset_no_summary_<stem>.json
"""

from __future__ import annotations
import json, random, sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Hashable, List, Tuple, Sequence, Optional

# ───────────────────────── CONFIG ───────────────────────── #
# Option 1: explicit list of sequence files
SEQ_JSONS: Sequence[Path] = [
    # Path("sequence_models/data/random_walk_two_step_red_green_goal.json"),
    # Path("sequence_models/data/random_walk_two_step_blue_purple_goal.json"),
]

# Option 2: directory with many *.json sequence files
SEQ_DIR: Optional[Path] = Path("sequence_models/data/two_color/random_walks")  # or None

# Shared dataset for all sequences
DATASET_JSON: Path = Path("sequence_models/data/four_flags_merged.json")

# Explicit output list (same length as SEQ_JSONS). Leave empty when using SEQ_DIR.
OUTPUT_JSONS: Sequence[Optional[Path]] = []

# Auto-naming for SEQ_JSONS (if OUTPUT_JSONS is empty)
OUTPUT_DIR: Optional[Path] = Path("sequence_models/data/two_color/no_summary")  # or None
FILENAME_PREFIX: str = "dataset_no_summary_"
FILENAME_SUFFIX: str = ""

RNG_SEED: Optional[int] = 42
# ─────────────────────────────────────────────────────────── #

DEAD_STATE: str = "dead"
EMBEDDING_DIM: int = 1024

def _sample_one(state: Hashable,
                dataset: Dict[Hashable, List[Dict[str, Any]]]
               ) -> Dict[str, Any]:
    if state == DEAD_STATE:
        return {"response": "", "embedding": [0.0] * EMBEDDING_DIM}
    pool = dataset.get(state, [])
    if not pool:
        raise ValueError(f"Dataset entry for state {state!r} is empty or missing")
    rec = random.choice(pool)
    d = {"response": rec["response"], "embedding": rec["embedding"]}
    if "subtask_decoder_label" in rec:
        d["subtask_decoder_label"] = rec["subtask_decoder_label"]
    return d

def extend_sequences(
    seq_list: List[Dict[str, List[Any]]],
    dataset: Dict[Hashable, List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    extended: List[Dict[str, Any]] = []
    for seq in seq_list:
        unique_states = set(seq["states"])
        choice = {s: _sample_one(s, dataset) for s in unique_states}

        responses = [choice[s]["response"] for s in seq["states"]]
        embeddings = [choice[s]["embedding"] for s in seq["states"]]
        have_labels = any("subtask_decoder_label" in choice[s] for s in unique_states)
        if have_labels:
            subtask_decoder_labels = [choice[s].get("subtask_decoder_label") for s in seq["states"]]

        new_seq = deepcopy(seq)
        new_seq["responses"] = responses
        new_seq["embeddings"] = embeddings
        if have_labels:
            new_seq["subtask_decoder_labels"] = subtask_decoder_labels
        extended.append(new_seq)
    return extended

def _load_json(p: Path) -> Any:
    with p.open(encoding="utf-8") as f:
        return json.load(f)

def _save_json(p: Optional[Path], obj: Any) -> None:
    if p is None:
        json.dump(obj, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
        print(f"[INFO] Extended sequences written to {p.resolve()}")

def _derive_inputs() -> List[Path]:
    if SEQ_DIR is not None:
        return sorted(SEQ_DIR.glob("*.json"))
    if SEQ_JSONS:
        return list(SEQ_JSONS)
    raise ValueError("No input specified: set SEQ_DIR or SEQ_JSONS")

def _derive_outputs(seq_paths: Sequence[Path]) -> List[Optional[Path]]:
    if SEQ_DIR is not None:
        # auto-name into OUTPUT_DIR
        if OUTPUT_DIR is None:
            raise ValueError("When using SEQ_DIR you must set OUTPUT_DIR")
        outs = []
        for sp in seq_paths:
            name = f"{FILENAME_PREFIX}{sp.stem}{FILENAME_SUFFIX}.json"
            outs.append(OUTPUT_DIR / name)
        return outs

    if OUTPUT_JSONS:
        if len(OUTPUT_JSONS) != len(seq_paths):
            raise ValueError("OUTPUT_JSONS length must match SEQ_JSONS length")
        return list(OUTPUT_JSONS)

    if OUTPUT_DIR is None:
        return [None] * len(seq_paths)

    outs = []
    for sp in seq_paths:
        name = f"{FILENAME_PREFIX}{sp.stem}{FILENAME_SUFFIX}.json"
        outs.append(OUTPUT_DIR / name)
    return outs

def main() -> None:
    if RNG_SEED is not None:
        random.seed(RNG_SEED)
    seq_paths = _derive_inputs()
    if not seq_paths:
        raise ValueError("No sequence files found")
    dataset = _load_json(DATASET_JSON)
    outputs = _derive_outputs(seq_paths)
    for seq_path, out_path in zip(seq_paths, outputs):
        seq_list = _load_json(seq_path)
        extended = extend_sequences(seq_list, dataset)
        _save_json(out_path, extended)

if __name__ == "__main__":
    main()
