#!/usr/bin/env python
"""
Merge several JSON sources into one file whose top level looks like:

{
  "N":  [ …records… ],          # from plain-list file
  "E":  [ …records… ],
  "F":  [ …records… ],
  "P1": [ …records from 'small' list inside p.json… ],
  "P2": [ …records from 'wide'  list inside p.json… ]
}
"""

import json
import pathlib
import sys
from typing import Dict, List, Tuple, Optional, Any

# ────────────────────── CUSTOMISE HERE ────────────────────── #

#   key : (file_path, sub_key or None)
KEY_TO_SOURCE: Dict[str, Tuple[str, Optional[str]]] = {
    "N":  ("sequence_models/data/language_data_complete_navigation.json",   None),
    "E":  ("sequence_models/data/language_data_complete_exploration.json", None),
    "F":  ("sequence_models/data/language_data_complete_idle.json",    None),

    # special file that contains {"small":[…], "wide":[…]}
    "P1": ("sequence_models/data/language_data_complete_flag_defense.json", "small"),   # becomes key "P1"
    "P2": ("sequence_models/data/language_data_complete_flag_defense.json", "wide"),    # becomes key "P2"
}

OUTPUT_FILE = "sequence_models/data/merged.json"

# ──────────────────────── UTILITIES ───────────────────────── #

def load_json(path: pathlib.Path) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        sys.exit(f"✗ {path}: {exc}")

def extract_records(src: Any, sub_key: Optional[str], path: pathlib.Path) -> List[dict]:
    """
    • If sub_key is None → src must already be a list.
    • If sub_key is given → src must be a dict and src[sub_key] must be a list.
    """
    if sub_key is None:
        if not isinstance(src, list):
            sys.exit(f"✗ {path}: expected a list at top level")
        return src

    # sub_key case
    if not isinstance(src, dict):
        sys.exit(f"✗ {path}: expected an object with key '{sub_key}'")
    if sub_key not in src or not isinstance(src[sub_key], list):
        sys.exit(f"✗ {path}: key '{sub_key}' missing or not a list")
    return src[sub_key]

# ─────────────────────────── MAIN ─────────────────────────── #

def main() -> None:
    cache: Dict[pathlib.Path, Any] = {}          # avoid reading same file twice
    result: Dict[str, List[dict]] = {}

    for key, (file_name, sub_key) in KEY_TO_SOURCE.items():
        path = pathlib.Path(file_name)
        if not path.exists():
            sys.exit(f"✗ file not found for key '{key}': {path}")

        # load (with cache)
        if path not in cache:
            cache[path] = load_json(path)

        records = extract_records(cache[path], sub_key, path)
        result[key] = records

    dest = pathlib.Path(OUTPUT_FILE)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"✓ merged {len(result)} keys → {dest.resolve()}")

if __name__ == "__main__":
    main()
