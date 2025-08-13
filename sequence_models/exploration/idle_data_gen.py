#!/usr/bin/env python3
"""
generate_idle_prompts.py
------------------------
Collect **N** “stay idle” prompts and save them as a JSON array.

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
python generate_idle_prompts.py -n 50

# 250 prompts into a custom file
python generate_idle_prompts.py -n 250 -o my_idle_prompts.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

# --------------------------- phrase banks ------------------------------

_LEADERS: List[str] = [
    "All units",
    "Expedition team",
    "Recon squad",
    "Exploration swarm",
    "Systems collective",
    "Task force",
    "Survey cohort",
    "Recovery detachment",
]

_IDLE_VERBS: List[str] = [
    "remain",
    "stay",
    "hold",
    "maintain",
    "keep",
    "stand",
]

_IDLE_ADVERBS: List[str] = [
    "idle",
    "stationary",
    "motionless",
    "in place",
    "on standby",
    "immobile",
]

_DURATIONS: List[str] = [
    "until further notice",
    "until the next directive",
    "until reactivation command",
    "until clearance is granted",
    "for the moment",
    "for now",
]

_GUIDELINES: List[str] = [
    "Continue passive sensor sweeps for situational awareness.",
    "Minimize power draw by suspending non-essential subsystems.",
    "Maintain network heartbeat every 30 seconds.",
    "Log environmental data but refrain from locomotion.",
    "Report anomalies immediately via secure channel.",
    "Keep vision stacks active to detect potential hazards.",
]

# --------------------------- public interface --------------------------

def stay_idle_prompt() -> str:
    """Generate a natural-language command instructing robots to hold position."""
    lines: List[str] = []

    # Core command
    lines.append(
        f"{random.choice(_LEADERS)}, {random.choice(_IDLE_VERBS)} "
        f"{random.choice(_IDLE_ADVERBS)} {random.choice(_DURATIONS)}."
    )

    # # Add two random guidelines for richness
    # lines.extend(random.sample(_GUIDELINES, k=2))

    return " ".join(lines)


def collect_prompts(n: int) -> list[dict[str, str]]:
    """Generate *n* distinct prompts wrapped in {'response': ...} objects."""
    return [{"response": stay_idle_prompt()} for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N stay-idle prompts and dump to JSON."
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
        default=Path("sequential_tasks/sentences/idle_sentences.json"),
        help='output JSON file (default: "sequential_tasks/sentences/idle_sentences.json")',
    )

    args = parser.parse_args()

    prompts = collect_prompts(args.num)

    # Write the list as a compact, UTF-8 encoded JSON array
    args.output.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.num} prompts → {args.output.resolve()}")


if __name__ == "__main__":
    main()
