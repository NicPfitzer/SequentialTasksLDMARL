#!/usr/bin/env python3
"""
generate_flag_defense_prompts.py
--------------------------------
Create natural-language commands telling robots to defend or flock
around a flag, with either a **small/tight** or **wide/large** formation.

Output JSON shape
-----------------
{
  "small": [
    {"response": "...", "radius": 0},
    ...
  ],
  "wide": [
    {"response": "...", "radius": 1},
    ...
  ]
}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

# ------------------------- constants & indices --------------------------

RADIUS_SMALL_INDEX = 0
RADIUS_LARGE_INDEX = 1

# ----------------------------- phrase banks -----------------------------

_LEADERS: List[str] = [
    "All units",
    "Expedition team",
    "Recon squad",
    "Task force",
    "Defense cohort",
    "Autonomous sentinels",
    "Guardian swarm",
]

_FLAG_NOUNS: List[str] = [
    "flag",
    "standard",
    "beacon",
    "marker",
    "rally point",
]

_DEFEND_VERBS: List[str] = [
    "defend",
    "guard",
    "protect",
    "secure",
    "shield",
    "hold",
]

_FLOCK_VERBS: List[str] = [
    "flock around",
    "circle",
    "surround",
    "orbit",
    "form up around",
    "gather round",
]

_SMALL_ADJS: List[str] = [
    "tight",
    "compact",
    "small",
    "close",
    "narrow",
]

_WIDE_ADJS: List[str] = [
    "wide",
    "broad",
    "large",
    "expanded",
    "extended",
]

_GUIDELINES: List[str] = [
    "Keep threat-detection sensors at maximum sensitivity.",
    "Use synchronized spacing to maintain formation integrity.",
    "Communicate adversary sightings instantly over the mesh network.",
    "Rotate front-line positions every 60 seconds to equalize load.",
    "Maintain 270-degree visual coverage of the perimeter.",
]

# --------------------------- prompt builders ----------------------------

def _formation_clause(size_adj: str) -> str:
    """Return a phrase like 'in a tight circle' or 'in a broad perimeter'."""
    if size_adj in _SMALL_ADJS:
        shape = random.choice(["circle", "cluster", "ring"])
    else:
        shape = random.choice(["perimeter", "arc", "circle"])
    return f"in a {size_adj} {shape}"

def _core_command(size_adj: str) -> str:
    """Build the main sentence (leader + verb + noun + formation)."""
    leader = random.choice(_LEADERS)
    flag = random.choice(_FLAG_NOUNS)
    if random.random() < 0.5:
        verb = random.choice(_DEFEND_VERBS)
        return f"{leader}, {verb} the {flag} {_formation_clause(size_adj)}."
    else:
        verb = random.choice(_FLOCK_VERBS)
        return f"{leader}, {verb} the {flag} {_formation_clause(size_adj)}."

def _full_prompt(size_adj: str) -> str:
    """Construct the full prompt, with one guideline for context."""
    lines = [_core_command(size_adj)]

    # One guideline for richness
    lines.append(random.choice(_GUIDELINES))

    # lines.append(random.choice(_GUIDELINES))  # ← extra guideline (commented out)

    return " ".join(lines)

# --------------------------- collection logic ---------------------------

def collect_prompts(n: int) -> dict[str, list[dict[str, str | int]]]:
    """Return {'small': [...], 'wide': [...]} with *n* prompts in each list."""
    return {
        "small": [
            {
                "response": _full_prompt(random.choice(_SMALL_ADJS)),
                "radius": RADIUS_SMALL_INDEX,
            }
            for _ in range(n)
        ],
        "wide": [
            {
                "response": _full_prompt(random.choice(_WIDE_ADJS)),
                "radius": RADIUS_LARGE_INDEX,
            }
            for _ in range(n)
        ],
    }

# -------------------------------- CLI -----------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N flag-defense prompts (small & wide) and dump to JSON."
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1000,
        help="number of prompts to generate per category (default: 1000)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("sequential_tasks/sentences/flag_defense_sentences.json"),
        help='output JSON file (default: "sequential_tasks/sentences/flag_defense_sentences.json")',
    )
    args = parser.parse_args()

    prompts = collect_prompts(args.num)

    args.output.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Wrote {args.num} small + {args.num} wide prompts "
        f"→ {args.output.resolve()}"
    )

if __name__ == "__main__":
    main()
