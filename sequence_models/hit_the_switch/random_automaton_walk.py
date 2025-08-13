"""merged_random_walks.py
---------------------------------
Run identical random‑walk experiments for multiple DFAs.

⚙ **Centralised experiment parameters**
   Edit *once* at the top (``MIN_STEPS``, ``MAX_STEPS``, ``REPS``, ``RNG_SEED``) and they apply to *all* DFAs.

Other features stay the same:
    • Each DFA described once in a lightweight `DFAConfig`.
    • Every record still has `events`, `states`, `success`.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

# --------------------------------------------------------------------------- #
#  Global experiment parameters – tweak them here
# --------------------------------------------------------------------------- #

MIN_STEPS: int = 2   # inclusive lower bound for random sequence length L
MAX_STEPS: int = 4  # inclusive upper bound for L
REPS: int = 500        # generate REPS successes + REPS failures per DFA
RNG_SEED: int | None = 42  # None ➟ do not reset RNG


# --------------------------------------------------------------------------- #
#  Generic DFA machinery
# --------------------------------------------------------------------------- #
NUM_AUTOMATA: int = 1  # number of automata in this experiment
EVENTS: Sequence[str] = ("a")              # index 0,1,2 ↔ a,b,c
Vector = List[int]                                      # e.g. [0, 1, 0]
State = str
Success = bool
Trace = Tuple[Vector, State, Vector]                    # [(vec, next_state), …]


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

    # ----------------------- core DFA API -----------------------

    def step(self, state: State, vec: Vector) -> Tuple[State]:

        return self._transition[state](vec) 
    
    def run(self, vectors: Sequence[Vector]) -> List[Trace]:
        state: State = self._initial
        trace: List[Trace] = [( None, state, self.state_to_one_hot(state))] 
        for v in vectors:
            state = self.step(state, v)
            state_one_hot = self.state_to_one_hot(state)
            trace.append((v, state, state_one_hot))
        return trace

    def state_to_one_hot(self, state: State) -> Vector:
        """Convert a state to its one-hot encoding."""
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
    return [random.randint(0, 1) for _ in EVENTS]


def random_walk(
    automaton: Automaton,
    steps: int,
) -> Trace:
    vecs = [rand_vec() for _ in range(steps)]
    return automaton.run(vecs)

# JSON conversion -----------------------------------------------------------

def record_trace(trace: List[State | Tuple[Vector, Vector]]) -> dict:
    events = [vec for (vec, _, _) in trace[1:]]
    states = [st for (_, st, _) in trace]
    label = [lb for (_,_, lb) in trace]
    return {"events": events, "states": states, "label": label}

# --------------------------------------------------------------------------- #
#  Mini‑DSL: describe each DFA in one @dataclass (no experiment params here!)
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
#  Concrete DFA definitions
# --------------------------------------------------------------------------- #
def  hit_the_switch_config() -> DFAConfig:
    # States:  S → G
    def next_S(v):
        a, = v
        return "G" if a else "S"

    def next_G(v):
        return "G"


    return DFAConfig(
        name="hit_the_switch",
        automaton_id=0,
        initial="S",
        finals={"G"},
        transition={"S": next_S, "G": next_G},
        outfile=Path("sequence_models/data/random_walk_hit_the_switch_results.json"),
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
        dfa.failed_transition = False
        trace = random_walk(dfa, L)
        all_runs.append(record_trace(trace))

    cfg.outfile.parent.mkdir(parents=True, exist_ok=True)
    cfg.outfile.write_text(json.dumps(all_runs, indent=2))
    print(f"[{cfg.name}] Wrote {len(all_runs)} runs ➜ {cfg.outfile.resolve()}")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    for cfg in (hit_the_switch_config(),):
        run_and_save(cfg)
