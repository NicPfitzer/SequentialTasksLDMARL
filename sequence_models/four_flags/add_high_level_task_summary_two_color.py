#!/usr/bin/env python
"""
Summarize two-color sequences from a directory using Gemini + Sentence-BERT, then
merge into a single JSON with minimal memory.

• Reads all *.json in INPUT_DIR (your "no_summary" dir).
• For each file, infers COLOR_1 -> COLOR_2 -> GOAL and auto-builds a prompt.
• Streams results into OUTPUT_JSON (resumable).
• Optional sentence override (global or per-file) for testing.

Set GOOGLE_API_KEY in your environment (or use GEMINI_API_KEY import).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import ijson
from ijson.common import IncompleteJSONError
from google import genai
from sentence_transformers import SentenceTransformer

# ─────────────── Project-local key import (keep as in your codebase) ───────── #
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sequence_models.api_keys import GEMINI_API_KEY  # noqa: E402

# ========================== Configuration ==========================

# Input directory with all no-summary files to process
INPUT_DIR = Path("sequence_models/data/one_color/no_summary")  # <- your no_summary dir
INPUT_GLOB = "dataset_no_summary_*.json"  # tweak if needed

# Output file (merged)
OUTPUT_JSON = Path("sequence_models/data/dataset_one_color_ALL.json")

# Gemini + embedding model
MODEL_NAME = "gemini-2.0-flash-lite"
TEMP = 0.2
MAX_PROMPT_CHARS = 24000
PRINT_EVERY = 200

# LLM client + embedder
#genai_client = genai.Client(api_key=GEMINI_API_KEY)
llm = SentenceTransformer("thenlper/gte-large")
# llm = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Optional testing overrides:
# 1) Global override for ALL items (string); set to None to disable.
SENTENCE_OVERRIDE_GLOBAL: Optional[str] = None
# 2) Per-file override (keyed by input filename stem -> sentence)
# SENTENCE_OVERRIDE_BY_FILE: Dict[str, str] = {
#     "dataset_no_summary_random_walk_two_step_red_green_goal":   "Locate the red flag, then the green one, and advance to the goal.",
#     "dataset_no_summary_random_walk_two_step_red_blue_goal":    "Find the red flag, spot the blue flag, and head for the target.",
#     "dataset_no_summary_random_walk_two_step_red_purple_goal":  "Identify the red flag, then the purple flag, and proceed to the goal.",

#     "dataset_no_summary_random_walk_two_step_green_red_goal":   "Search for the green flag, then the red flag, and navigate to the goal.",
#     "dataset_no_summary_random_walk_two_step_green_blue_goal":  "Locate the green flag, then the blue flag, and move toward the target.",
#     "dataset_no_summary_random_walk_two_step_green_purple_goal":"Find the green flag, identify the purple one, then reach the goal.",

#     "dataset_no_summary_random_walk_two_step_blue_red_goal":    "Spot the blue flag, then the red flag, and continue to the goal.",
#     "dataset_no_summary_random_walk_two_step_blue_green_goal":  "Head for the blue flag, then the green flag, and finish at the target.",
#     "dataset_no_summary_random_walk_two_step_blue_purple_goal": "Identify the blue flag, then the purple flag, and advance to the goal.",

#     "dataset_no_summary_random_walk_two_step_purple_red_goal":  "Locate the purple flag, then the red flag, and proceed to the goal.",
#     "dataset_no_summary_random_walk_two_step_purple_green_goal":"Find the purple flag, then the green one, and head toward the target.",
#     "dataset_no_summary_random_walk_two_step_purple_blue_goal": "Search for the purple flag, then the blue flag, and reach the goal.",
# }
SENTENCE_OVERRIDE_BY_FILE: Dict[str, str] = {
    "dataset_no_summary_random_walk_red_goal":   "Locate the red flag, then the switch, and advance to the goal.",
    "dataset_no_summary_random_walk_blue_goal":    "Find the red flag, spot the switch, and head for the target.",
    "dataset_no_summary_random_walk_purple_goal":  "Identify the purple flag, navigate to the switch and proceed to the goal.",
    "dataset_no_summary_random_walk_green_goal":   "Search for the green flag, then the switch, and navigate to the goal.",
}

# ========================== Utilities ==========================

COLORS = ("red", "green", "blue", "purple")

def already_processed() -> int:
    """Count complete array items already written to OUTPUT_JSON (resume-safe)."""
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
    files = sorted(INPUT_DIR.glob(INPUT_GLOB))
    if not files:
        raise FileNotFoundError(f"No input files matched {INPUT_GLOB} in {INPUT_DIR}")
    return files

# ---------- Color inference ----------

FNAME_PATTERN = re.compile(
    r"(?:two_step_)?(?P<c1>red|green|blue|purple)_(?P<c2>red|green|blue|purple)_goal",
    re.IGNORECASE,
)

def infer_colors_from_filename(p: Path) -> Optional[Tuple[str, str]]:
    m = FNAME_PATTERN.search(p.stem)
    if not m:
        return None
    c1, c2 = m.group("c1").lower(), m.group("c2").lower()
    if c1 in COLORS and c2 in COLORS:
        return c1, c2
    return None

def infer_colors_from_states(seq_first_items: Sequence[dict]) -> Optional[Tuple[str, str]]:
    """Fallback: scan the earliest two distinct color states in the stream."""
    seen_order: List[str] = []
    for item in seq_first_items:
        for st in item.get("states", []):
            if st in COLORS:
                if not seen_order or seen_order[-1] != st:
                    seen_order.append(st)
                if len(seen_order) >= 2:
                    return seen_order[0], seen_order[1]
    return None

# ---------- Prompt building ----------

def build_prompt_template(color1: str, color2: str) -> str:
    return textwrap.dedent(f"""
        Write ONE short imperative sentence (≤16 words) that states:
        first locate the {color1} flag, then the {color2} flag, then go to the goal.
        Use varied vocabulary across outputs: try different verbs (find/locate/identify/spot, reach/head to/navigate to),
        and vary sentence structure. Return only the sentence, starting with a capital and ending with a period.
    """).strip()


# def one_sentence_summary(responses: List[str], prompt_tmpl: str) -> str:
#     bullet_responses = [f"- {r}" for r in responses]
#     joined = "\n".join(bullet_responses)

#     # Truncate if too long for the model
#     base_prompt = prompt_tmpl.format(joined="{joined}")
#     max_joined_chars = MAX_PROMPT_CHARS - len(base_prompt.format(joined=""))
#     if len(joined) > max_joined_chars:
#         truncated = joined[-max_joined_chars:]
#         truncated = "\n".join(truncated.split("\n")[1:])  # avoid partial bullet
#     else:
#         truncated = joined

#     prompt = prompt_tmpl.format(joined=truncated)
#     response = genai_client.models.generate_content(
#         model=MODEL_NAME,
#         contents=[prompt],
#         config={"temperature": TEMP},
#     )
#     return response.text

# ========================== Processing ==========================

def stream_process():
    files = list_input_files()

    processed_global = already_processed()
    skipped = 0

    mode = "a" if processed_global else "w"
    with OUTPUT_JSON.open(mode, encoding="utf-8") as dst:
        if mode == "w":
            dst.write("[\n")
            first_entry = True
        else:
            dst.seek(dst.tell() - 2, os.SEEK_SET)  # remove trailing "\n]\n"
            dst.truncate()
            first_entry = False

        processed = processed_global
        start_ts = time.time()

        for f_idx, input_path in enumerate(files):
            # Determine override (if any) for this file
            file_override = SENTENCE_OVERRIDE_BY_FILE.get(input_path.stem)
            if file_override:
                print(f"[INFO] Using per-file override for {input_path.name}")
            use_global_override = SENTENCE_OVERRIDE_GLOBAL is not None

            # Determine colors
            colors = infer_colors_from_filename(input_path)
            if colors is None:
                # Peek a few items to infer from states
                with input_path.open("rb") as src:
                    items = ijson.items(src, "item")
                    peek = []
                    try:
                        for _ in range(10):  # small peek window
                            peek.append(next(items))
                    except StopIteration:
                        pass
                colors = infer_colors_from_states(peek)
                if colors is None:
                    raise ValueError(f"Could not infer two colors for {input_path.name}")
            c1, c2 = colors

            # Build prompt template
            prompt_template = build_prompt_template(c1, c2)

            # Now stream the whole file
            with input_path.open("rb") as src:
                items = ijson.items(src, "item")
                for seq in items:
                    if skipped < processed_global:
                        skipped += 1
                        continue

                    # Decide the sentence to use
                    if use_global_override:
                        sentence = SENTENCE_OVERRIDE_GLOBAL
                    elif file_override:
                        sentence = file_override
                    # else:
                    #     sentence = one_sentence_summary(seq["responses"], prompt_template)

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
