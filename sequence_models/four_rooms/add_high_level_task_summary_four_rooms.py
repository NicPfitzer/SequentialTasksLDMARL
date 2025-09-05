#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import ijson
from ijson.common import IncompleteJSONError
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

INPUT_DIR = Path("sequence_models/data/four_rooms/no_summary")
INPUT_GLOB = "dataset_no_summary_*.json"
OUTPUT_JSON = Path("sequence_models/data/four_rooms/dataset_four_rooms.json")
PRINT_EVERY = 200

llm = SentenceTransformer("thenlper/gte-large")

SENTENCE_OVERRIDE_GLOBAL: Optional[str] = None
SENTENCE_OVERRIDE_BY_FILE: Dict[str, str] = {}

AGENTS = ["Units", "Agents", "Robots", "Recon agents", "Exploration drones", "Navigation bots", "Swarm members"]
VERBS = ["open", "unlock", "activate"]
ADVANCE = ["then proceed", "then advance", "then move", "then head"]
GOAL_PHRASES = ["to the goal room", "into room four", "to the final room", "to the objective room"]
CLARIFIERS = [
    "from left to right",
    "across the three gates",
    "in sequence",
    "in order",
    "starting at the left-most room",
]

def choose(xs: List[str]) -> str:
    import random
    return random.choice(xs)

def global_objective_sentence() -> str:
    import random
    agent = choose(AGENTS)
    v = choose(VERBS)
    adv = choose(ADVANCE)
    goal = choose(GOAL_PHRASES)
    clar = choose(CLARIFIERS)
    variants = [
        f"{agent}, {v} the first, second, and third switches {clar}, {adv} {goal} and reach the target.",
        f"{agent}, {v} three switches in order ({clar}); {adv} {goal} and complete the objective.",
        f"{agent}, sequentially {v} switches one, two, and three {clar}, {adv} {goal} and finish at the goal.",
        f"{agent}, {v} the three room switches {clar}; {adv} {goal} and finalize at the goal.",
        f"{agent}, {v} switches 1–3 {clar}, {adv} {goal} and conclude at the goal.",
    ]
    s = choose(variants)
    return " ".join(s.split())

def already_processed() -> int:
    if not OUTPUT_JSON.exists():
        return 0
    cnt = 0
    with OUTPUT_JSON.open("rb") as f:
        try:
            for prefix, event, _ in ijson.parse(f):
                if (prefix, event) == ("item", "start_map"):
                    cnt += 1
        except IncompleteJSONError:
            pass
    return cnt

def list_input_files() -> List[Path]:
    files = sorted(INPUT_DIR.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No input files matched {INPUT_GLOB} in {INPUT_DIR}")
    return files

def sentence_for_item(input_path: Path) -> str:
    if SENTENCE_OVERRIDE_GLOBAL is not None:
        return SENTENCE_OVERRIDE_GLOBAL
    file_override = SENTENCE_OVERRIDE_BY_FILE.get(input_path.stem)
    if file_override:
        return file_override
    return global_objective_sentence()

def stream_process():
    files = list_input_files()
    processed_global = already_processed()
    skipped = 0
    mode = "a" if processed_global else "w"
    with OUTPUT_JSON.open(mode, encoding="utf-8") as dst:
        if mode == "w":
            dst.write("[\n")
            first_entry = True
        else:
            dst.seek(dst.tell() - 2, os.SEEK_SET)
            dst.truncate()
            first_entry = False
        processed = processed_global
        start_ts = time.time()
        for input_path in files:
            with input_path.open("rb") as src:
                items = ijson.items(src, "item")
                for seq in items:
                    if skipped < processed_global:
                        skipped += 1
                        continue
                    sentence = sentence_for_item(input_path)
                    seq["summary"] = sentence
                    seq["y"] = llm.encode(sentence, convert_to_numpy=True).tolist()
                    if "embeddings" in seq:
                        seq["h"] = [[float(x) for x in row] for row in seq["embeddings"]]
                        seq.pop("embeddings", None)
                    if not first_entry:
                        dst.write(",\n")
                    else:
                        first_entry = False
                    json.dump(seq, dst, ensure_ascii=False, indent=2)
                    processed += 1
                    if processed % PRINT_EVERY == 0:
                        elapsed = time.time() - start_ts
                        print(f"[{processed}] processed  elapsed {elapsed:.1f}s")
        dst.write("\n]\n")
    elapsed = time.time() - start_ts
    print(f"\n[INFO] Finished – wrote {processed} items to {OUTPUT_JSON.resolve()}")
    print(f"[INFO] Total time: {elapsed:.1f}s | Avg/item: {elapsed / max(processed,1):.2f}s")

if __name__ == "__main__":
    stream_process()
