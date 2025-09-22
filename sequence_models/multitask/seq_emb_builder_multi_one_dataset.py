#!/usr/bin/env python
"""
seq_emb_builder_multi_dir.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
• Uses ONE shared dataset for ALL sequences.
• Input options:
    – List of explicit sequence files (SEQ_JSONS), or
    – One directory or multiple directories containing *.json sequence files (SEQ_DIR).
• Writes one output per input. If SEQ_DIR is used, outputs go to OUTPUT_DIR
  with auto names: dataset_no_summary_<stem>.json
"""

from __future__ import annotations
import json, random, sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Hashable, List, Tuple, Sequence, Optional, Union

# ───────────────────────── CONFIG ───────────────────────── #
# Option 1: explicit list of sequence files
SEQ_JSONS: Sequence[Path] = [
    # Path("sequence_models/data/random_walk_two_step_red_green_goal.json"),
    # Path("sequence_models/data/random_walk_two_step_blue_purple_goal.json"),
]

# Option 2: directory (Path) OR list/tuple of directories with *.json sequence files
SEQ_DIR: Optional[Union[Path, Sequence[Path]]] = [Path("sequence_models/data/one_color/random_walks/"), Path("sequence_models/data/capture_flag/random_walks/"), Path("sequence_models/data/four_rooms/random_walks/")]  # or None or [Path(...), Path(...)]

# Shared dataset for all sequences
DATASET_JSON: Path = Path("sequence_models/data/multitask_merged.json")

# Explicit output list (same length as SEQ_JSONS). Leave empty when using SEQ_DIR.
OUTPUT_JSONS: Sequence[Optional[Path]] = []

# Auto-naming for SEQ_JSONS (if OUTPUT_JSONS is empty) or when using SEQ_DIR.
# If SEQ_DIR is a list, this must be a list of equal length.
OUTPUT_DIR: Optional[Union[Path, Sequence[Path]]] = [
    Path("sequence_models/data/one_color/no_summary"),
    Path("sequence_models/data/capture_flag/no_summary"),
    Path("sequence_models/data/four_rooms/no_summary"),
]
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

def _as_path_list(x: Union[Path, Sequence[Path]]) -> List[Path]:
    if isinstance(x, (list, tuple)):
        return [Path(d) for d in x]
    return [Path(x)]

def _derive_inputs_outputs() -> tuple[List[Path], List[Optional[Path]]]:
    if SEQ_DIR is not None:
        dirs = _as_path_list(SEQ_DIR)
        outs = _as_path_list(OUTPUT_DIR) if OUTPUT_DIR is not None else None
        if outs is None or len(outs) != len(dirs):
            raise ValueError("When using SEQ_DIR as a list, OUTPUT_DIR must be a list of the same length")
        seq_paths, out_paths = [], []
        for d, out_dir in zip(dirs, outs):
            files = sorted(d.glob("*.json"))
            for f in files:
                seq_paths.append(f)
                out_paths.append(out_dir / f"{FILENAME_PREFIX}{f.stem}{FILENAME_SUFFIX}.json")
        return seq_paths, out_paths

    if SEQ_JSONS:
        seq_paths = list(SEQ_JSONS)
        if OUTPUT_JSONS:
            if len(OUTPUT_JSONS) != len(seq_paths):
                raise ValueError("OUTPUT_JSONS length must match SEQ_JSONS length")
            return seq_paths, list(OUTPUT_JSONS)
        if OUTPUT_DIR is None:
            return seq_paths, [None] * len(seq_paths)
        outs = []
        out_dirs = _as_path_list(OUTPUT_DIR)
        if len(out_dirs) != 1:
            raise ValueError("For SEQ_JSONS, OUTPUT_DIR must be a single path")
        for sp in seq_paths:
            outs.append(out_dirs[0] / f"{FILENAME_PREFIX}{sp.stem}{FILENAME_SUFFIX}.json")
        return seq_paths, outs

    raise ValueError("No input specified: set SEQ_DIR or SEQ_JSONS")

def main() -> None:
    if RNG_SEED is not None:
        random.seed(RNG_SEED)
    seq_paths, outputs = _derive_inputs_outputs()
    if not seq_paths:
        raise ValueError("No sequence files found")
    dataset = _load_json(DATASET_JSON)
    for seq_path, out_path in zip(seq_paths, outputs):
        seq_list = _load_json(seq_path)
        extended = extend_sequences(seq_list, dataset)
        _save_json(out_path, extended)

if __name__ == "__main__":
    main()