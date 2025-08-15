#!/usr/bin/env python3
"""
generate_prompts_navigate_to_object.py
--------------------------------------
Collect **N** 'navigate to the <color> <flag-like object>' prompts
and save them as JSON arrays — one file per color.

Colors: red, green, blue, purple

Output format
-------------
[
  {"response": "prompt 1"},
  {"response": "prompt 2"},
  ...
]

Usage examples
--------------
# 50 prompts per color into default directory
python generate_prompts_navigate_to_object.py -n 50

# 250 prompts per color into a custom directory
python generate_prompts_navigate_to_object.py -n 250 -d data/sentences
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

# ----- Lexicons -----

_AGENTS: List[str] = [
    "Units",
    "Agents",
    "Robots",
    "Support bots",
    "Team leaders",
    "Recon agents",
    "Autonomous nodes",
    "Task units",
    "Exploration drones",
    "Navigation bots",
    "Control units",
    "Swarm members",
    "Sentinel units",
]

_VERBS: List[str] = [
    "navigate",
    "move",
    "proceed",
    "head",
    "advance",
    "travel",
    "go",
    "reach",
    "approach",
    "make your way",
    "get to",
    "find your way",
    "locate",
]

_ADVERBS: List[str] = [
    "quickly",
    "carefully",
    "directly",
    "steadily",
    "efficiently",
    "immediately",
    "without hesitation",
    "swiftly",
    "smoothly",
    "promptly",
    "without delay",
    "expeditiously",
    "securely",
]

# Common, banner/marker sense only
FLAG_OBJECTS: List[str] = [
    "flag",
    "banner",
    "ensign",
    "pennant",
    "emblem",
    "symbol",
    "marker",
    "badge",
    "tag",
    "totem",
    "token",
]

COLORS: List[str] = ["red", "green", "blue", "purple"]


# ----- Prompt generation -----

def navigate_to_object_prompt(color: str) -> str:
    """Generate a natural-language command instructing units to move to a colored flag-like object."""
    agent = random.choice(_AGENTS)
    verb = random.choice(_VERBS)
    adverb = random.choice(_ADVERBS)
    obj = random.choice(FLAG_OBJECTS)

    # Some adverbs sound better after the object
    post_object_adverbs = {"without delay", "without hesitation", "expeditiously"}
    if adverb in post_object_adverbs:
        return f"{agent}, {verb} to the {color} {obj} {adverb}."
    else:
        return f"{agent}, {verb} {adverb} to the {color} {obj}."


def collect_prompts(n: int, color: str) -> list[dict[str, str]]:
    return [{"response": navigate_to_object_prompt(color)} for _ in range(n)]


# ----- CLI -----

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N navigation-to-<color>-flag prompts and dump one JSON per color."
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1000,
        help="number of prompts to generate per color (default: 1000)",
    )
    parser.add_argument(
        "-d",
        "--outdir",
        type=Path,
        default=Path("sentences"),
        help='output directory for JSON files (default: "sentences")',
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional RNG seed for reproducibility",
    )

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)

    for color in COLORS:
        prompts = collect_prompts(args.num, color)
        out_file = args.outdir / f"navigate_to_{color}_flag.json"
        out_file.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.num} prompts → {out_file.resolve()}")

if __name__ == "__main__":
    main()
