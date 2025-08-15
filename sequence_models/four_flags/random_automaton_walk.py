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

# --------------------------------------------------------------------------- #
#  Global experiment parameters – tweak them here
# --------------------------------------------------------------------------- #

MIN_STEPS: int = 3  # inclusive lower bound for sequence length L
MAX_STEPS: int = 15  # inclusive upper bound for L
REPS: int = 2000      # number of runs per DFA
RNG_SEED: int | None = 42  # None ➟ do not reset RNG

# --------------------------------------------------------------------------- #
#  Generic DFA machinery
# --------------------------------------------------------------------------- #

NUM_AUTOMATA: int = 1

# Event vector: [R, G, B, P, SWITCH, GOAL]
EVENTS: Sequence[str] = (
    "found_red_flag",
    "found_green_flag",
    "found_blue_flag",
    "found_purple_flag",
    "found_switch",
    "found_goal",
)

R, G, B, P, SW, GL = range(6)

Vector = List[int]               # e.g. [1,0,0,0,0,0]
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

    def state_to_one_hot(self, state: State) -> List[int]:
        state_one_hot = [0] * (len(self._transition))
        index = list(self._transition.keys()).index(state)
        state_one_hot[index] = 1
        automaton_one_hot = [0] * NUM_AUTOMATA
        automaton_one_hot[self.automaton_id] = 1
        return automaton_one_hot + state_one_hot


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


# --------------------------------------------------------------------------- #
#  Concrete DFA: find RGBP, then SWITCH, then GOAL, then STAY at GOAL
# --------------------------------------------------------------------------- #

def find_colors_then_switch_then_goal_config() -> DFAConfig:
    """
    States (short & indicative):
      red     → waiting for green
      green   → waiting for blue
      blue    → waiting for purple
      purple  → waiting for switch
      switch  → waiting for goal
      goal    → at goal (before stay)
    """

    def advance_on(bit_idx: int, next_state: State, else_state: State) -> Callable[[Vector], State]:
        return lambda v, idx=bit_idx, ns=next_state, es=else_state: (ns if v[idx] else es)

    transition: Dict[State, Callable[[Vector], State]] = {
        "red":    advance_on(G,  "green",  "red"),
        "green":  advance_on(B,  "blue",   "green"),
        "blue":   advance_on(P,  "purple", "blue"),
        "purple": advance_on(SW, "switch", "purple"),
        "switch": advance_on(GL, "goal",   "switch"),
        "goal":   (lambda v: "goal"),
    }

    return DFAConfig(
        name="find_rgbp_switch_goal",
        automaton_id=0,
        initial="red",
        finals={"goal"},
        transition=transition,
        outfile=Path("sequence_models/data/random_walk_rgbp_switch_goal_results.json"),
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
        L = random.randint(MIN_STEPS, MAX_STEPS)
        trace = random_walk(dfa, L)
        all_runs.append(record_trace(trace))

    cfg.outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg.outfile.write_text(json.dumps(all_runs, indent=2))
    print(f"[{cfg.name}] Wrote {len(all_runs)} runs ➜ {cfg.outfile.resolve()}")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    for cfg in (find_colors_then_switch_then_goal_config(),):
        run_and_save(cfg)
