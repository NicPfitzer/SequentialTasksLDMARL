"""merged_random_walks.py
---------------------------------
Run identical random‑walk experiments for multiple DFAs.

⚙ Centralised experiment parameters
   Edit once at the top (MIN_STEPS, MAX_STEPS, REPS, RNG_SEED).

Every record has: events, states, label (automaton+state one-hot per step).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple
import math

# --------------------------------------------------------------------------- #
#  Global experiment parameters – tweak them here
# --------------------------------------------------------------------------- #

MIN_SEQ_STEPS: int = 3  # inclusive lower bound for sequence length L
MAX_SEQ_STEPS: int = 12  # inclusive upper bound for L
REPS: int = 1000      # number of runs per DFA
RNG_SEED: int | None = 42  # None ➟ do not reset RNG

EVENT_DIM = 5
NUM_AUTOMATA= 3

FIND_RED = 0
FIND_GREEN = 1
FIND_BLUE = 2
FIND_PURPLE = 3
FIND_SWITCH = 4
FIND_GOAL = 5

STATES = {
    "FIND_GOAL": FIND_GOAL,
    "goal": FIND_GOAL,
    "FIND_SWITCH": FIND_SWITCH,
    "switch": FIND_SWITCH,
    "FIND_RED": FIND_RED,
    "red": FIND_RED,
    "FIND_GREEN": FIND_GREEN,
    "green": FIND_GREEN,
    "FIND_BLUE": FIND_BLUE,
    "blue": FIND_BLUE,
    "FIND_PURPLE": FIND_PURPLE,
    "purple": FIND_PURPLE,
}



# --------------------------------------------------------------------------- #
#  Generic DFA machinery
# --------------------------------------------------------------------------- #

NUM_AUTOMATA: int = 4

# Event vector: [R, G, B, P, SWITCH]
EVENTS: Sequence[str] = (
    "found_red_flag",
    "found_green_flag",
    "found_blue_flag",
    "found_purple_flag",
    "found_switch",
)

R, G, B, P, SW = range(5)
idx_map = {"red": R, "green": G, "blue": B, "purple": P}

Vector = List[int]               # e.g. [1,0,0,0,0]
State = str
Trace = Tuple[Vector | None, State, List[int]]

class Automaton:
    def __init__(
        self,
        transition: Dict[State, Callable[[Vector], State]],
        initial: State,
        finals: set[State],
        automaton_id: int,
    ) -> None:
        self._transition = transition
        self._initial = initial
        self._finals = finals
        self.automaton_id = automaton_id

    def step(self, state: State, vec: Vector) -> State:
        return self._transition[state](vec)

    def run(self, vectors: Sequence[Vector]) -> List[Trace]:
        state: State = self._initial
        trace: List[Trace] = [(None, state, self.state_to_one_hot(state))]
        for v in vectors:
            state = self.step(state, v)
            state_one_hot = self.state_to_one_hot(state)
            trace.append((v, state, state_one_hot))
        return trace

    def _to_bits(self, n: int, width: int) -> List[int]:
        # MSB first
        return [(n >> i) & 1 for i in reversed(range(width))] if width > 0 else [0]

    def state_to_one_hot(self, state: State) -> List[int]:
        # State one-hot (unchanged)
        state_one_hot = [0] * (len(EVENTS) + 1)
        index = STATES[state]
        state_one_hot[index] = 1

        # Automaton binary code (replaces one-hot)
        # width = ceil(log2(NUM_AUTOMATA)); use at least 1 bit even if only one automaton
        width = max(1, math.ceil(math.log2(NUM_AUTOMATA)))
        automaton_bits = self._to_bits(self.automaton_id, width)

        return automaton_bits + state_one_hot


# --------------------------------------------------------------------------- #
#  Experiment helpers
# --------------------------------------------------------------------------- #

def rand_vec() -> Vector:
    # independent Bernoulli(0.5) bits; tweak if you want sparser hits
    return [random.randint(0, 1) for _ in EVENTS]

def random_walk(automaton: Automaton, steps: int) -> List[Trace]:
    vecs = [rand_vec() for _ in range(steps)]
    return automaton.run(vecs)

def record_trace(trace: List[Trace]) -> dict:
    events = [vec for (vec, _, _) in trace[1:]]
    states = [st for (_, st, _) in trace]
    label = [lb for (_, _, lb) in trace]
    return {"events": events, "states": states, "label": label}


# --------------------------------------------------------------------------- #
#  Mini‑DSL: describe each DFA in one @dataclass
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DFAConfig:
    name: str
    automaton_id: int
    initial: State
    finals: set[State]
    transition: Dict[State, Callable[[Vector], State]]
    outfile: Path


# ─── Transition Predicates ───────────────────────────────────────────────────────── #

from typing import Callable, Iterable, Tuple

Pred = Callable[[Vector], bool]

def has(bit_idx: int) -> Pred:
    return lambda v, i=bit_idx: bool(v[i])

def not_has(bit_idx: int) -> Pred:
    return lambda v, i=bit_idx: not v[i]

def all_of(*preds: Pred) -> Pred:
    return lambda v, ps=preds: all(p(v) for p in ps)

def any_of(*preds: Pred) -> Pred:
    return lambda v, ps=preds: any(p(v) for p in ps)

def advance_if(pred: Pred, then_state: State, else_state: State) -> Callable[[Vector], State]:
    """Single guard: if pred(v) then then_state else else_state."""
    return lambda v, p=pred, t=then_state, e=else_state: (t if p(v) else e)

def advance_case(*cases: Tuple[Pred, State], default: State) -> Callable[[Vector], State]:
    """
    Ordered guards: for the first (pred, state) where pred(v) is True, go to state.
    Otherwise go to `default`.
    """
    def step(v: Vector, cs=cases, d=default) -> State:
        for pred, st in cs:
            if pred(v):
                return st
        return d
    return step

# --------------------------------------------------------------------------- #
#  Concrete DFA: find RGBP, then SWITCH, then GOAL, then STAY at GOAL
# --------------------------------------------------------------------------- #

def find_rgbd_then_switch_then_goal_config() -> DFAConfig:
    """Find flags in RGBP order, hit the switch, and navigate to the goal."""

    transition: Dict[State, Callable[[Vector], State]] = {
        "red":     advance_if(has(R),  "green",  "red"),
        "green":   advance_if(has(G),  "blue",   "green"),
        "blue":    advance_if(has(B),  "purple", "blue"),
        "purple":  advance_if(has(P),  "switch", "purple"),
        "switch":  advance_if(has(SW), "goal",   "switch"),
        "goal":    (lambda v: "goal"),
    }

    return DFAConfig(
        name="find_rgbp_switch_goal",
        automaton_id=0,
        initial="red",
        finals={"goal"},
        transition=transition,
        outfile=Path("sequence_models/data/random_walk_four_flags_rgbp.json"),
    )

def find_bgrp_then_switch_then_goal_config() -> DFAConfig:
    """Find flags in BGRP order, hit the switch, and navigate to the goal."""
    
    transition: Dict[State, Callable[[Vector], State]] = {
        "blue":   advance_if(has(B),  "switch", "blue"),
        "switch": advance_if(has(SW), "goal",   "switch"),
        "goal":   (lambda v: "goal"),
    }

    return DFAConfig(
        name="find_blue_switch_goal",
        automaton_id=1,
        initial="blue",
        finals={"goal"},
        transition=transition,
        outfile=Path("sequence_models/data/random_walk_four_flags_blue.json"),
    )

def switch_then_goal_config() -> DFAConfig:
    """Find flags in BGRP order, hit the switch, and navigate to the goal."""

    transition: Dict[State, Callable[[Vector], State]] = {
        "switch": advance_if(has(SW), "goal", "switch"),
        "goal":   (lambda v: "goal"),
    }

    return DFAConfig(
        name="find_switch_goal",
        automaton_id=2,
        initial="switch",
        finals={"goal"},
        transition=transition,
        outfile=Path("sequence_models/data/random_walk_four_flags_switch_goal.json"),
    )

# --------------------------------------------------------------------------- #
#  Concrete DFA generator: COLOR -> SWITCH -> GOAL (order matters)
# --------------------------------------------------------------------------- #

def make_color_then_goal_config(color: State, automaton_id: int) -> DFAConfig:
    """Generic two-step color sequence then goal."""

    transition: Dict[State, Callable[[Vector], State]] = {
        color: advance_case(
            (all_of(has(idx_map[color]), has(SW)), "goal"),
            (has(idx_map[color]), "switch"),
            default=color,
        ),
        "switch": advance_case(
            (has(SW), "goal"),
            (not_has(idx_map[color]), color),
            default="switch"
        ),
        "goal":   (lambda v: "goal"),
    }

    name = f"find_{color}_goal"
    outfile = Path(f"sequence_models/data/one_color/random_walks/random_walk_{color}_goal.json")

    return DFAConfig(
        name=name,
        automaton_id=automaton_id,
        initial=color,
        finals={"goal"},
        transition=transition,
        outfile=outfile,
    )

def color_then_goal_configs(start_id: int = 0) -> List[DFAConfig]:
    """All ordered pairs without repetition: 4 * 3 = 12."""
    colors = ["red", "green", "blue", "purple"]
    cfgs: List[DFAConfig] = []
    k = 0
    for c1 in colors:
            cfgs.append(make_color_then_goal_config(c1, start_id + k))
            k += 1
    return cfgs

# --------------------------------------------------------------------------- #
#  Concrete DFA generator: COLOR_1 -> COLOR_2 -> -> SWITCH -> GOAL (order matters)
# --------------------------------------------------------------------------- #

def make_two_color_then_goal_config(color1: State, color2: State, automaton_id: int) -> DFAConfig:
    """Generic two-step color sequence then goal."""
    idx_map = {"red": R, "green": G, "blue": B, "purple": P}

    transition: Dict[State, Callable[[Vector], State]] = {
        color1:   advance_if(has(idx_map[color1]), color2,   color1),
        color2:   advance_if(has(idx_map[color2]), "switch", color2),
        "switch": advance_if(has(SW),              "goal",   "switch"),
        "goal":   (lambda v: "goal"),
    }

    name = f"find_{color1}_{color2}_goal"
    outfile = Path(f"sequence_models/data/two_color/random_walks/random_walk_two_step_{color1}_{color2}_goal.json")

    return DFAConfig(
        name=name,
        automaton_id=automaton_id,
        initial=color1,
        finals={"goal"},
        transition=transition,
        outfile=outfile,
    )


def two_color_then_goal_configs(start_id: int = 0) -> List[DFAConfig]:
    """All ordered pairs without repetition: 4 * 3 = 12."""
    colors = ["red", "green", "blue", "purple"]
    cfgs: List[DFAConfig] = []
    k = 0
    for c1 in colors:
        for c2 in colors:
            if c1 == c2:
                continue  # keep order-specific, no repetition
            cfgs.append(make_two_color_then_goal_config(c1, c2, start_id + k))
            k += 1
    return cfgs

# --------------------------------------------------------------------------- #
#  Master runner (uses GLOBAL parameters)
# --------------------------------------------------------------------------- #

def run_and_save(cfg: DFAConfig) -> None:
    if RNG_SEED is not None:
        random.seed(RNG_SEED)

    dfa = Automaton(cfg.transition, cfg.initial, cfg.finals, cfg.automaton_id)

    all_runs: List[dict] = []
    for _ in range(REPS):
        L = random.randint(MIN_SEQ_STEPS, MAX_SEQ_STEPS)
        trace = random_walk(dfa, L)
        all_runs.append(record_trace(trace))

    cfg.outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg.outfile.write_text(json.dumps(all_runs, indent=2))
    print(f"[{cfg.name}] Wrote {len(all_runs)} runs ➜ {cfg.outfile.resolve()}")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    
    # AUTOMATA = [
    #     find_rgbd_then_switch_then_goal_config(),
    #     find_bgrp_then_switch_then_goal_config(),
    #     switch_then_goal_config(),
    # ]
    
    #AUTOMATA = []
    #AUTOMATA += two_color_then_goal_configs(start_id=len(AUTOMATA))
    
    AUTOMATA = []
    AUTOMATA += color_then_goal_configs(start_id=len(AUTOMATA))
    NUM_AUTOMATA: int = len(AUTOMATA)
    
    for cfg in AUTOMATA:
        run_and_save(cfg)
