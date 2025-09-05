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
REPS: int = 3000      # number of runs per DFA
RNG_SEED: int | None = 42  # None ➟ do not reset RNG

EVENT_DIM = 3
NUM_AUTOMATA= 1

FIND_FIRST_SWITCH = 0
FIND_SECOND_SWITCH = 1
FIND_THIRD_SWITCH = 2
FIND_GOAL = 3

STATES = {
    "FIND_GOAL": FIND_GOAL,
    "goal": FIND_GOAL,
    "FIND_FIRST_SWITCH": FIND_FIRST_SWITCH,
    "first": FIND_FIRST_SWITCH,
    "FIND_SECOND_SWITCH": FIND_SECOND_SWITCH,
    "second": FIND_SECOND_SWITCH,
    "FIND_THIRD_SWITCH": FIND_THIRD_SWITCH,
    "third": FIND_THIRD_SWITCH,
}



# --------------------------------------------------------------------------- #
#  Generic DFA machinery
# --------------------------------------------------------------------------- #

NUM_AUTOMATA: int = 4

# Event vector: [F, S, T, G]
EVENTS: Sequence[str] = (
    "found_first_switch",
    "found_second_switch",
    "found_third_switch",
)

F, S, T, G = range(4)
idx_map = {"first": F, "second": S, "third": T, "goal": G}

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

def hit_all_switches_then_goal(start_id: int) -> DFAConfig:
    """Find switches in order and navigate to the goal."""
    
    transition: Dict[State, Callable[[Vector], State]] = {
        "first":     advance_if(has(F),  "second",  "first"),
        "second":   advance_if(has(S),  "third",   "second"),
        "third":    advance_if(has(T),  "goal", "third"),
        "goal":    (lambda v: "goal"),
    }

    return DFAConfig(
        name="hit_all_switches_then_goal",
        automaton_id=start_id,
        initial="first",
        finals={"goal"},
        transition=transition,
        outfile=Path("sequence_models/data/four_rooms/random_walk_four_rooms.json"),
    )


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
    AUTOMATA += [hit_all_switches_then_goal(start_id=len(AUTOMATA))]
    NUM_AUTOMATA: int = len(AUTOMATA)
    
    for cfg in AUTOMATA:
        run_and_save(cfg)
