#!/usr/bin/env python3
"""
generate_prompts_navigate_to_goal.py
------------------------------------
Collect **N** 'navigate to the goal/objective' prompts and save them as a JSON array.

Output format
-------------
[
  {"response": "prompt 1"},
  {"response": "prompt 2"},
  ...
]

Usage examples
--------------
# 50 prompts to the default file
python generate_prompts_navigate_to_goal.py -n 50

# 250 prompts into a custom file
python generate_prompts_navigate_to_goal.py -n 250 -o my_prompts.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

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

_OBJECTS: List[str] = [
    "final goal",
    "final objective",
    "ultimate target",
    "primary mission objective",
    "designated final goal",
    "mission’s final objective",
    "assigned ultimate target",
    "final destination",
    "operation’s final goal",
    "strategic end objective",
    "mission endpoint",
    "final checkpoint",
    "target zone for completion",
]

def navigate_to_objective_prompt() -> str:
    """Generate a natural-language command instructing a unit to move to the final goal/objective."""
    agent = random.choice(_AGENTS)
    verb = random.choice(_VERBS)
    adverb = random.choice(_ADVERBS)
    obj = random.choice(_OBJECTS)

    return f"{agent}, {verb} {adverb} to the {obj}."

def collect_prompts(n: int) -> list[dict[str, str]]:
    return [{"response": navigate_to_objective_prompt()} for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N navigation-to-goal prompts and dump to JSON."
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1000,
        help="number of prompts to generate (default: 1000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sentences/navigate_to_goal.json"),
        help='output JSON file (default: "sentences/navigate_to_goal.json")',
    )

    args = parser.parse_args()

    prompts = collect_prompts(args.num)

    args.output.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.num} prompts → {args.output.resolve()}")


if __name__ == "__main__":
    main()
