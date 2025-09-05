#!/usr/bin/env python3
"""
generate_prompts_navigate_to_object.py
--------------------------------------
Collect **N** 'navigate to the switch/button' prompts and save them as a JSON array.

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
python generate_prompts_navigate_to_object.py -n 50

# 250 prompts into a custom file
python generate_prompts_navigate_to_object.py -n 250 -o my_prompts.json
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
    "switch",
    "button",
    "main control button",
    "override switch",
    "shutdown button",
    "activation switch",
    "access button",
    "control switch",
    "power button",
    "command switch",
    "system control button",
    "interface switch",
    "activation button",
    "system switch",
]

def navigate_to_object_prompt() -> str:
    """Generate a natural‑language command instructing a unit to move to an object."""
    return (
        f"{random.choice(_AGENTS)}, {random.choice(_VERBS)} {random.choice(_ADVERBS)} "
        f"to the {random.choice(_OBJECTS)}."
    )


def collect_prompts(n: int) -> list[dict[str, str]]:
    return [{"response": navigate_to_object_prompt()} for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N navigation-to-switch prompts and dump to JSON."
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
        default=Path("sentences/navigate_to_switch.json"),
        help='output JSON file (default: "sentences/navigate_to_switch.json")',
    )

    args = parser.parse_args()

    prompts = collect_prompts(args.num)

    args.output.write_text(
        json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {args.num} prompts → {args.output.resolve()}")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
generate_prompts_navigate_to_switch.py
--------------------------------------
Collect N 'navigate to the <ordinal> switch' prompts and save them as JSON arrays
— one file per ordinal.

Ordinals: first, second, third

We make each prompt distinctive by combining:
- Clear ordinal: first, second, third
- Room info: left-most room, second room from the left, third room before the goal
- Spatial cues: near the divider gate, close to the left wall, centered in the bay
- Gate context: the gate between rooms 0-1, 1-2, 2-3
- Varying agent nouns, verbs, and adverbs

Output format
-------------
[
  {"response": "prompt 1"},
  {"response": "prompt 2"},
  ...
]

Usage examples
--------------
# 50 prompts per ordinal into default directory
python generate_prompts_navigate_to_switch.py -n 50

# 250 prompts per ordinal into a custom directory
python generate_prompts_navigate_to_switch.py -n 250 -d data/sentences
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict

# ----- Lexicons -----

AGENTS: List[str] = [
    "Units",
    "Agents",
    "Robots",
    "Support bots",
    "Recon agents",
    "Autonomous nodes",
    "Task units",
    "Exploration drones",
    "Navigation bots",
    "Control units",
    "Swarm members",
    "Sentinel units",
]

VERBS: List[str] = [
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

ADVERBS: List[str] = [
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

SWITCH_NOUNS: List[str] = [
    "switch",
    "toggle",
    "activation pad",
    "release plate",
    "trigger",
    "control switch",
    "unlock switch",
]

ROOM_LEFT_PHRASES: List[str] = [
    "in the left-most room",
    "in room 0 on the far left",
    "in the first room from the left",
    "in the starting bay on the left side",
]
ROOM_MID_PHRASES: List[str] = [
    "in the second room from the left",
    "in room 1, past the first gate",
    "in the mid-left room",
    "in the room after the left-most bay",
]
ROOM_RIGHT_PHRASES: List[str] = [
    "in the third room, just before the goal room",
    "in room 2, ahead of the final gate",
    "in the pre-goal room on the right",
    "in the last room before the goal",
]

NEAR_GATE_PHRASES = [
    "near the divider gate",
    "close to the vertical gate",
    "beside the gate post",
    "by the room divider",
]

WALL_CUES = [
    "hug the left wall on entry",
    "center yourself and proceed",
    "scan along the back wall",
    "follow the floor markers down the middle",
    "keep to the inner lane",
    "track the right wall briefly, then cut in",
]

CONFIRM_CLAUSES = [
    "confirm activation when the indicator lights",
    "stop once the actuator clicks",
    "mark success when the plate depresses",
    "acknowledge once the panel turns on",
    "signal ready when the ring glows",
]

# Gate context between rooms i and i+1 for extra clarity
GATE_CONTEXT: Dict[int, List[str]] = {
    0: [
        "the gate between rooms 0 and 1",
        "the first divider gate",
        "the initial barrier",
    ],
    1: [
        "the gate between rooms 1 and 2",
        "the middle divider gate",
        "the second barrier",
    ],
    2: [
        "the gate between rooms 2 and 3",
        "the last divider before the goal room",
        "the final barrier",
    ],
}

# Switch metadata
SWITCHES = [
    {
        "ordinal": "first",
        "room_index": 0,
        "room_phrases": ROOM_LEFT_PHRASES,
        "gate_phrases": GATE_CONTEXT[0],
    },
    {
        "ordinal": "second",
        "room_index": 1,
        "room_phrases": ROOM_MID_PHRASES,
        "gate_phrases": GATE_CONTEXT[1],
    },
    {
        "ordinal": "third",
        "room_index": 2,
        "room_phrases": ROOM_RIGHT_PHRASES,
        "gate_phrases": GATE_CONTEXT[2],
    },
]

TEMPLATES: List[str] = [
    # Each template must contain {agent} {verb} {adverb} {ordinal} {switch_noun}
    # and may include {room_phrase}, {gate_phrase}, {near_gate}, {wall_cue}, {confirm}
    "{agent}, {verb} {adverb} to the {ordinal} {switch_noun} {room_phrase} near {gate_phrase}; {confirm}.",
    "{agent}, {verb} {adverb} toward the {ordinal} {switch_noun} {room_phrase}, {near_gate}.",
    "{agent}, {verb} {adverb} to the {ordinal} {switch_noun} {room_phrase}. Then {confirm}.",
    "{agent}, {verb} {adverb} to the {ordinal} {switch_noun} located by {gate_phrase} {room_phrase}.",
    "{agent}, {verb} to the {ordinal} {switch_noun} {adverb} {room_phrase}; {near_gate}, and {confirm}.",
    "{agent}, {verb} {adverb} to the {ordinal} {switch_noun} {room_phrase}. On approach, {wall_cue}.",
    "{agent}, {verb} {adverb} until you reach the {ordinal} {switch_noun} {room_phrase} beside {gate_phrase}.",
    "{agent}, {verb} {adverb} to the {ordinal} {switch_noun} {room_phrase}. Stay aligned and {confirm}.",
    "{agent}, {verb} {adverb} straight to the {ordinal} {switch_noun} {room_phrase}, right next to {gate_phrase}.",
    "{agent}, {verb} {adverb} to the {ordinal} {switch_noun} {room_phrase}. Approach {near_gate} and {confirm}.",
]

POST_OBJECT_ADVERBS = {"without delay", "without hesitation", "expeditiously"}

def _choose(seq: List[str]) -> str:
    return random.choice(seq)

def _maybe_commas(text: str) -> str:
    # Clean double spaces and awkward spaces before punctuation
    return " ".join(text.split()).replace(" ,", ",")

def prompt_for_switch(sw_meta: Dict) -> str:
    agent = _choose(AGENTS)
    verb = _choose(VERBS)
    adverb = _choose(ADVERBS)
    switch_noun = _choose(SWITCH_NOUNS)

    ordinal = sw_meta["ordinal"]
    room_phrase = _choose(sw_meta["room_phrases"])
    gate_phrase = _choose(sw_meta["gate_phrases"])
    near_gate = _choose(NEAR_GATE_PHRASES)
    wall_cue = _choose(WALL_CUES)
    confirm = _choose(CONFIRM_CLAUSES)

    # Some adverbs read better after the object
    if adverb in POST_OBJECT_ADVERBS:
        # Move adverb post-object by tweaking verb chunk slightly in template
        adverb_chunk = adverb
        # Pick a template and inject variables
        template = _choose(TEMPLATES)
        s = template.format(
            agent=agent,
            verb=verb,
            adverb=adverb_chunk,
            ordinal=ordinal,
            switch_noun=switch_noun,
            room_phrase=room_phrase,
            gate_phrase=gate_phrase,
            near_gate=near_gate,
            wall_cue=wall_cue,
            confirm=confirm,
        )
    else:
        template = _choose(TEMPLATES)
        s = template.format(
            agent=agent,
            verb=verb,
            adverb=adverb,
            ordinal=ordinal,
            switch_noun=switch_noun,
            room_phrase=room_phrase,
            gate_phrase=gate_phrase,
            near_gate=near_gate,
            wall_cue=wall_cue,
            confirm=confirm,
        )

    return _maybe_commas(s)

def collect_prompts(n: int, sw_meta: Dict) -> List[Dict[str, str]]:
    return [{"response": prompt_for_switch(sw_meta)} for _ in range(n)]

# ----- CLI -----

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate N navigation-to-<ordinal>-switch prompts and write one JSON per ordinal."
    )
    parser.add_argument(
        "-n",
        "--num",
        type=int,
        default=1000,
        help="number of prompts to generate per ordinal (default: 1000)",
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

    for sw in SWITCHES:
        prompts = collect_prompts(args.num, sw)
        out_file = args.outdir / f"navigate_to_{sw['ordinal']}_switch.json"
        out_file.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {args.num} prompts → {out_file.resolve()}")

if __name__ == "__main__":
    main()
