#!/usr/bin/env python
from __future__ import annotations

import json
import os
import re
import sys
import time
import textwrap
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Literal, Sequence, Union

import ijson
from ijson.common import IncompleteJSONError
from sentence_transformers import SentenceTransformer

# ───────── project-local import (unchanged) ───────── #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ========================== Configuration ==========================

# Accept a single Path OR a list of Paths
INPUT_DIR: Union[Path, Sequence[Path]] = [
    Path("sequence_models/data/one_color/no_summary"),
    Path("sequence_models/data/capture_flag/no_summary"),
    Path("sequence_models/data/four_rooms/no_summary"),
]
INPUT_GLOB = "*.json"
OUTPUT_JSON = Path("sequence_models/data/dataset_multitask.json")

PRINT_EVERY = 200

# Set to int for reproducibility, or None for full randomness
RNG_SEED: Optional[int] = 42

llm = SentenceTransformer("thenlper/gte-large")

# Overrides (exact stem or canonical stem); used verbatim if present
SENTENCE_OVERRIDE_BY_FILE: Dict[str, str] = {
    # Goal tasks
    "random_walk_red_goal":    "Locate the red flag, then the switch, and advance to the goal.",
    "random_walk_green_goal":  "Search for the green flag, then the switch, and navigate to the goal.",
    "random_walk_blue_goal":   "Find the blue flag, spot the switch, and head for the target.",
    "random_walk_purple_goal": "Identify the purple flag, navigate to the switch and proceed to the goal.",

    # Defend & return tasks
    "random_walk_find_flag_defend":        "Explore the area for the flag, then defend the position from the opposing team.",
    "random_walk_find_flag_navigate_home": "Search the area to locate the flag and bring it back to the base.",
    
    # Four room switch task
    "random_walk_four_rooms": "Agents, sequentially activate switches one, two, and three from left to right, then proceed to the goal."
}

# ========================== Utilities ==========================

COLORS = ("red", "green", "blue", "purple")

_CANON_PREFIXES = ("dataset_no_summary_", "no_summary_", "ns_", "ns__")
_CANON_SUFFIXES = ("", "_nosum")

def _canonical_stem(stem: str) -> str:
    s = stem
    changed = True
    while changed:
        changed = False
        for pre in _CANON_PREFIXES:
            if s.startswith(pre):
                s = s[len(pre):]
                changed = True
    for suf in _CANON_SUFFIXES:
        if suf and s.endswith(suf):
            s = s[: -len(suf)]
    return s

def _as_dir_list(dirs: Union[Path, Sequence[Path]]) -> List[Path]:
    if isinstance(dirs, (list, tuple)):
        return [Path(d) for d in dirs]
    return [Path(dirs)]

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
    dirs = _as_dir_list(INPUT_DIR)
    files_set = set()
    for d in dirs:
        files = d.glob(INPUT_GLOB)
        for p in files:
            files_set.add(p.resolve())
    files_sorted = sorted(files_set)  # deterministic order across dirs
    if not files_sorted:
        raise FileNotFoundError(f"No input files matched {INPUT_GLOB} in {', '.join(str(d) for d in dirs)}")
    return [Path(p) for p in files_sorted]

# ---------- Task inference ----------

# goal-type: looks like "...<color>_goal"
RE_COLOR_GOAL = re.compile(r"(?P<color>red|green|blue|purple)_goal", re.IGNORECASE)

# defend-type markers
RE_DEFEND = re.compile(r"(defend|hold|secure(_position)?|guard)", re.IGNORECASE)

# navigate-home / return markers
RE_RETURN = re.compile(r"(navigate[_\-]?home|return[_\-]?home|bring[_\-]?back|back[_\-]?to[_\-]?base)", re.IGNORECASE)

# four-rooms switch task marker
RE_FOUR_ROOMS = re.compile(r"(four[_\-]?rooms|three[_\-]?switches|hit[_\-]?all[_\-]?switches)", re.IGNORECASE)

TaskType = Literal["goal", "defend", "return", "four_rooms"]

def infer_task_type_and_color(p: Path) -> Tuple[TaskType, Optional[str]]:
    stem = _canonical_stem(p.stem)

    m_goal = RE_COLOR_GOAL.search(stem)
    if m_goal:
        c = m_goal.group("color").lower()
        if c in COLORS:
            return "goal", c

    if RE_DEFEND.search(stem) or "defend" in stem:
        return "defend", None

    if RE_RETURN.search(stem) or "navigate_home" in stem or "nav_home" in stem or "return" in stem:
        return "return", None
    
    if RE_FOUR_ROOMS.search(stem) or "four_rooms" in stem or "fourrooms" in stem or "hit_all_switches" in stem:
        return "four_rooms", None

    raise ValueError(f"Could not infer task type from filename: {p.name}")

# ---------- Diversity sentence generators per task family ----------

def _choose(xs: List[str]) -> str:
    return random.choice(xs)

def _tidy(s: str) -> str:
    s = " ".join(s.split())
    if not s.endswith("."):
        s += "."
    return s[0].upper() + s[1:]

# 1) Flag → Switch → Goal (color-specific)
FIND_VERBS = ["Find", "Locate", "Identify", "Spot", "Search for"]
SWITCH_NOUNS = ["the switch", "the lever", "the control switch", "the actuator"]
THEN = ["then", "and then", "next"]
GOAL_VERBS = ["go to", "head to", "navigate to", "proceed to", "reach"]
GOAL_NOUNS = ["the goal", "the target", "the objective", "the final room", "the goal room"]

def sentence_goal(color: str) -> str:
    t1 = f"{_choose(FIND_VERBS)} the {color} flag, {_choose(THEN)} {_choose(SWITCH_NOUNS)}, {_choose(THEN)} {_choose(GOAL_VERBS)} {_choose(GOAL_NOUNS)}"
    t2 = f"First {_choose(FIND_VERBS).lower()} the {color} flag, {_choose(THEN)} {_choose(SWITCH_NOUNS)}, {_choose(THEN)} {_choose(GOAL_VERBS)} {_choose(GOAL_NOUNS)}"
    t3 = f"{_choose(FIND_VERBS)} the {color} flag; {_choose(THEN)} {_choose(SWITCH_NOUNS)}; {_choose(THEN)} {_choose(GOAL_VERBS)} {_choose(GOAL_NOUNS)}"
    return _tidy(_choose([t1, t2, t3]))

# 2) Find-and-Defend
SEARCH_VERBS = ["Search for", "Locate", "Find", "Scout for", "Seek"]
FLAG_OBJS = ["the flag", "the objective flag", "the mission flag"]
DEFEND_VERBS = ["defend", "hold", "secure", "protect", "guard"]
POSITION_NOUNS = ["the position", "the site", "the location", "the capture point", "the area"]
ADAPT_PHRASES = [
    "adjust tactics as needed",
    "adapt your defense",
    "respond to threats",
    "coordinate as required",
    "maintain situational awareness",
]

def sentence_defend() -> str:
    v1 = f"{_choose(SEARCH_VERBS)} {_choose(FLAG_OBJS)}, {_choose(THEN)} {_choose(DEFEND_VERBS)} {_choose(POSITION_NOUNS)}; {_choose(ADAPT_PHRASES)}"
    v2 = f"{_choose(SEARCH_VERBS)} {_choose(FLAG_OBJS)} and {_choose(DEFEND_VERBS)} {_choose(POSITION_NOUNS)}; {_choose(ADAPT_PHRASES)}"
    v3 = f"Retrieve {_choose(FLAG_OBJS)}, {_choose(THEN)} {_choose(DEFEND_VERBS)} {_choose(POSITION_NOUNS)} and {_choose(ADAPT_PHRASES)}"
    return _tidy(_choose([v1, v2, v3]))

# 3) Find-and-Return (navigate home)
RETURN_VERBS = ["return", "bring", "deliver", "carry", "escort"]
HOME_NOUNS = ["the base", "home base", "the spawn", "the starting zone", "your base"]
ROUTE_PHRASES = [
    "via a safe route",
    "avoiding hostiles",
    "while minimizing exposure",
    "using shortest paths",
    "maintaining cover",
]

def sentence_return() -> str:
    v1 = f"{_choose(SEARCH_VERBS)} {_choose(FLAG_OBJS)}, {_choose(THEN)} {_choose(RETURN_VERBS)} the flag to {_choose(HOME_NOUNS)} {_choose(ROUTE_PHRASES)}"
    v2 = f"Find {_choose(FLAG_OBJS)}, {_choose(THEN)} {_choose(RETURN_VERBS)} it back to {_choose(HOME_NOUNS)} {_choose(ROUTE_PHRASES)}"
    v3 = f"Locate {_choose(FLAG_OBJS)} and {_choose(RETURN_VERBS)} it to {_choose(HOME_NOUNS)} {_choose(ROUTE_PHRASES)}"
    return _tidy(_choose([v1, v2, v3]))

# 4) Hit all switches in order, then goal
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

def sentence_four_rooms() -> str:
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

def make_sentence(task: TaskType, color: Optional[str]) -> str:
    if task == "goal":
        if not color:
            raise ValueError("Goal task requires a color")
        return sentence_goal(color)
    if task == "defend":
        return sentence_defend()
    if task == "return":
        return sentence_return()
    if task == "four_rooms":
        return sentence_four_rooms()
    raise RuntimeError("Unknown task type")

def _resolve_override_for_file(p: Path) -> Optional[str]:
    stem = p.stem
    canon = _canonical_stem(stem)
    return SENTENCE_OVERRIDE_BY_FILE.get(stem) or SENTENCE_OVERRIDE_BY_FILE.get(canon)

# ========================== Processing ==========================

def stream_process():
    if RNG_SEED is not None:
        random.seed(RNG_SEED)

    files = list_input_files()
    processed_global = already_processed()
    skipped = 0

    mode = "a" if processed_global else "w"
    with OUTPUT_JSON.open(mode, encoding="utf-8") as dst:
        if mode == "w":
            dst.write("[\n")
            first_entry = True
        else:
            # remove trailing "\n]\n" to resume appending
            dst.seek(dst.tell() - 2, os.SEEK_SET)
            dst.truncate()
            first_entry = False

        processed = processed_global
        start_ts = time.time()

        for input_path in files:
            file_override = _resolve_override_for_file(input_path)
            if file_override:
                print(f"[INFO] Using per-file override for {input_path.name}")

            task, color = infer_task_type_and_color(input_path)

            with input_path.open("rb") as src:
                items = ijson.items(src, "item")
                for seq in items:
                    if skipped < processed_global:
                        skipped += 1
                        continue

                    # Use override if present; else generate per-item diverse sentence
                    sentence = make_sentence(task, color)

                    # Enrich + convert
                    seq["summary"] = sentence
                    seq["y"] = llm.encode(sentence, convert_to_numpy=True).tolist()
                    seq["h"] = [[float(x) for x in row] for row in seq["embeddings"]]
                    seq.pop("embeddings", None)

                    # Write
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

# ========================== Entrypoint ==========================

if __name__ == "__main__":
    stream_process()
