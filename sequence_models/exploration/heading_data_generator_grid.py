import os
import time
import json
import random
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from io import BytesIO
from PIL import Image
from tqdm import tqdm
from google import genai
from typing import Optional, Tuple

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_keys import GEMINI_API_KEY

# ========================== Configuration ==========================

# Gemini API Key
genai_client = genai.Client(api_key=GEMINI_API_KEY)  # 🔐 Replace with your actual key

# Dataset generation parameters
NUM_DATA_TARGETS = 1000
API_DELAY_SECONDS = 0.0
grid_size = 10
min_patch_size = 1
max_patch_size = 25
min_std = 0.05
max_std = 5.0
max_num_patches = 1
multipatch_prob = 0.5
no_patch_prob = 0.0
danger_zone_prob = 0.0
BATCH_SIZE = 1

# ========================== Vocabulary ==========================

color_dict = {
    "red":      {"rgb": [1.0, 0.0, 0.0], "index": 0},
    "green":    {"rgb": [0.0, 1.0, 0.0], "index": 1},
    "blue":     {"rgb": [0.0, 0.0, 1.0], "index": 2},
    "yellow":   {"rgb": [1.0, 1.0, 0.0], "index": 3},
    "orange":   {"rgb": [1.0, 0.5, 0.0], "index": 4},
    # "cyan":     {"rgb": [0.0, 1.0, 1.0], "index": 5},
    # "magenta":  {"rgb": [1.0, 0.0, 1.0], "index": 6},
    # "purple":   {"rgb": [0.5, 0.0, 0.5], "index": 7},
    # "pink":     {"rgb": [1.0, 0.75, 0.8], "index":8},
    # "brown":    {"rgb": [0.6, 0.4, 0.2], "index": 9},
    # "gray":     {"rgb": [0.5, 0.5, 0.5], "index": 10}
}

direction_terms = {
    "ern": ["eastern", "western", "southern", "northern", "center"],
    "cardinal": ["east", "west", "south", "north", "middle"],
    "erly": ["easterly", "westerly", "southerly", "norhterly", "center"],
    "dunno": ["upper", "lower", "leftmost", "rightmost", "middle"],
    "dunno_2": ["top", "bottom", "left", "right", "center"]
}

size_terms = {
    "cat_1": ["vast", "moderate", "tiny"],
    "cat_2": ["expansive", "average", "miniature"],
    "cat_3": ["immense", "intermediate", "compact"],
    "cat_4": ["enormous", "mid-sized", "narrow"],
    "cat_5": ["extensive", "medium-scale", "petite"],
    "cat_6": ["broad", "midsize", "modest"],
    "cat_7": ["wide", "standard", "limited"],
    "cat_8": ["colossal", "fair-sized", "restricted"],
    "cat_9": ["large", "medium", "small"]
}

environment_terms = [
    *["area", "region", "zone", "territory", "surroundings", "environment", "field", "landscape", "setting", "domain"],
    *[f"{prefix} {term}" for prefix in ["search", "exploration", "reconnaissance", "investigation"]
      for term in ["area", "region", "zone", "territory", "surroundings", "environment", "field", "landscape", "setting", "domain"]]
]

danger_zone_terms = [
    "danger zone", "danger region", "hot zone", "hot region", "red zone", "red region",
    "hazard area", "hazard region", "restricted area", "restricted region", "no-go zone",
    "no-go region", "kill zone", "kill region", "combat zone", "combat region", "war zone",
    "war region", "exclusion zone", "exclusion region", "critical zone", "critical region",
    "unsafe area", "unsafe region", "high-risk area", "high-risk region", "death zone",
    "death region", "threat zone", "threat region"
]

target_terms = [
    "Landmark", "Endpoint", "Reference point", "Objective", "Goal", "Mark", "Point", "Focus",
    "Destination", "Aim", "Spot", "Site", "Position", "Location", "Zone", "Subject", "Waypoint"
]

confidence_txt = [
    "very confident about the likely location of the targets",
    "reasonably confident about the general area but unsure of the exact spot",
    "uncertain and relying on sparse clues to guess the location"
]

tone_styles = [
    "Give a terse and tactical report, like a military commander.",
    "Use a friendly and encouraging tone, as if speaking to junior robots.",
    "Use formal and technical language, suitable for an engineering briefing.",
    "Speak like a seasoned explorer guiding a mission through uncertain terrain."
]

# ---------------- Vocabulary helper utilities ----------------

def maybe_use_vocab(default_word: str, vocab_list: list, prob: float = 0.3) -> str:
    """Return a random term from *vocab_list* with probability *prob*, otherwise *default_word*."""
    return random.choice(vocab_list) if random.random() < prob else default_word

# Pre‑baked synonym lists for the size buckets returned by *label_size*.
size_synonyms = {
    "very small": ["tiny", "miniature", "compact", "petite", "restricted"],
    "small": ["small", "compact", "limited", "narrow", "modest"],
    "medium-sized": ["medium", "average", "standard", "intermediate", "mid-sized", "midsize"],
    "large": ["large", "wide", "broad", "extensive", "immense"],
    "very large": ["very large", "vast", "enormous", "colossal", "immense", "expansive"]
}

def maybe_use_size_synonym(label: str, prob: float = 0.3) -> str:
    """Return a size synonym for *label* with probability *prob*."""
    return random.choice(size_synonyms[label]) if random.random() < prob else label

# ========================== Utility Functions ==========================

def label_size(ratio: float) -> str:
    if ratio < 0.05:  return "very small"
    if ratio < 0.10:  return "small"
    if ratio < 0.15:  return "medium-sized"
    if ratio < 0.20:  return "large"
    return "very large"

def compute_patch_centroid(grid: np.ndarray, patch_value: int = 1) -> Optional[Tuple[float, float]]:
    """Return the (row, col) centroid of the cells matching *patch_value*.
    If no such cells exist, return None."""
    coords = np.argwhere(grid == patch_value)
    if coords.size == 0:
        return None
    # Mean row, mean col – keep as float so we can convert to percentage later if needed
    centroid = coords.mean(axis=0)
    return tuple(centroid)

# ----------------------------------------------------------------------
# (The remainder of the file below this point is the original implementation,
#  with *describe_image* updated to call the new vocabulary helpers.)
# ----------------------------------------------------------------------

def get_neighbors(cell, grid_size):
    r, c = cell
    neighbors = []
    if r > 0: neighbors.append((r-1, c))
    if r < grid_size-1: neighbors.append((r+1, c))
    if c > 0: neighbors.append((r, c-1))
    if c < grid_size-1: neighbors.append((r, c+1))
    return neighbors

def generate_gaussian_prob_map(center, std_x, std_y, grid_size):
    cx, cy = center
    x = np.arange(grid_size)
    y = np.arange(grid_size)
    xx, yy = np.meshgrid(x, y, indexing='ij')
    gauss = np.exp(-((xx - cx) ** 2) / (2 * std_x ** 2) - ((yy - cy) ** 2) / (2 * std_y ** 2))
    return gauss / gauss.sum()

def plot_grid(grid, rgb):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='binary')
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

from matplotlib.patches import Rectangle

def plot_grid_with_rgb(grid, rgb):
    # Create an RGB image with all white
    H, W = grid.shape
    color_img = np.ones((H, W, 3), dtype=np.float32)

    # Apply the rgb color to the positions where grid == 1
    for i in range(3):
        color_img[:, :, i] = np.where(grid == 1, rgb[i], 1.0)

    # Plot
    fig, ax = plt.subplots()
    ax.imshow(color_img)
    
    grid_size_local = grid.shape[0]
    ax.set_xticks(np.arange(grid_size_local))
    ax.set_yticks(np.arange(grid_size_local))
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Add outer boundary rectangle
    rect = Rectangle((-0.45,-0.45), grid_size_local-0.15, grid_size_local-0.1,
                     linewidth=2, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # Save to buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)

    return buf

def plot_color_grid(grid):
    # Define custom colors: -1 (blue), 0 (white), 1 (red)
    colors = ['blue', 'white', 'red']
    cmap = ListedColormap(colors)

    # Shift grid values from [-1, 0, 1] to [0, 1, 2] to index into the colormap
    color_index_grid = (np.array(grid) + 1).astype(int)

    fig, ax = plt.subplots()
    ax.imshow(color_index_grid, cmap=cmap)
    ax.axis("off")
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    plt.close(fig)
    return buf

# ========================== Core Functions ==========================

# (generate_grid_target_danger_pair & generate_grid_and_patches unchanged)

def generate_grid_target_danger_pair():
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    used_cells = set()
    target_flag = True
    
    for patch_idx in range(max_num_patches):
        
        if patch_idx % 2 == 1: target_flag = False
        
        candidates = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(10)]
        best_start = max(candidates, key=lambda cell: min([np.linalg.norm(np.subtract(cell, u)) for u in used_cells], default=0))
        
        if best_start in used_cells:
                    continue

        prob_map = generate_gaussian_prob_map(
            center=best_start,
            std_x=random.uniform(min_std, max_std),
            std_y=random.uniform(min_std, max_std),
            grid_size=grid_size
        )

        patch = {best_start}
        frontier = set(get_neighbors(best_start, grid_size))
        target_size = random.randint(min_patch_size, max_patch_size)

        while len(patch) < target_size and frontier:
            probs = np.array([prob_map[r, c] if (r, c) not in used_cells else 0. for r, c in frontier])
            if probs.sum() == 0: break
            next_cell = random.choices(list(frontier), weights=probs, k=1)[0]
            patch.add(next_cell)
            frontier.remove(next_cell)
            frontier.update({n for n in get_neighbors(next_cell, grid_size) if n not in patch and n not in used_cells})

        value = 1 if target_flag else -1
        for r, c in patch:
            grid[r, c] = value
        used_cells.update(patch)
        
    return grid
    
def generate_grid_and_patches():
    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    used_cells = set()
    num_patches = 0
    target_flag = random.random() > danger_zone_prob
    
    color_info = []
    patch_sizes = []

    if random.random() > no_patch_prob:
        for patch_idx in range(max_num_patches):
            
            color_name, _color_info = random.choice(list(color_dict.items()))
            color_index = _color_info["index"]
            _color = _color_info["rgb"]
            num_targets = random.randint(1, 5)
            conf_level  = random.randint(0, 2)

            color_info.append({
                "name":  color_name,
                "index": color_index,
                "rgb":   _color,
                "num_targets": num_targets,
                "confidence":  conf_level
            })
            
            if patch_idx == 0 or random.random() < multipatch_prob:
                num_patches += 1
                candidates = [(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)) for _ in range(10)]
                best_start = max(candidates, key=lambda cell: min([np.linalg.norm(np.subtract(cell, u)) for u in used_cells], default=0))

                if best_start in used_cells:
                    continue

                prob_map = generate_gaussian_prob_map(
                    center=best_start,
                    std_x=random.uniform(min_std, max_std),
                    std_y=random.uniform(min_std, max_std),
                    grid_size=grid_size
                )

                patch = {best_start}
                frontier = set(get_neighbors(best_start, grid_size))
                target_size = random.randint(min_patch_size, max_patch_size)

                while len(patch) < target_size and frontier:
                    
                    probs = np.array([prob_map[r, c] if (r, c) not in used_cells else 0. for r, c in frontier])
                    if probs.sum() == 0: break
                    next_cell = random.choices(list(frontier), weights=probs, k=1)[0]
                    patch.add(next_cell)
                    frontier.remove(next_cell)
                    frontier.update({n for n in get_neighbors(next_cell, grid_size) if n not in patch and n not in used_cells})
                patch_sizes.append(len(patch))

                value = 1 if target_flag else -1
                for r, c in patch:
                    grid[r, c] = value
                used_cells.update(patch)

    return grid, num_patches, target_flag, color_info, patch_sizes

# ----------------------------------------------------------------------
# describe_image
# ----------------------------------------------------------------------

def describe_image(
    image_buf,
    *,
    grid_size: int,
    patch_size: int,
    color_name: Optional[str] = None,
    num_targets: Optional[int] = None,
    confidence: Optional[int] = None,
    include_color: bool = True,
    include_number: bool = True,
    include_confidence: bool = True,
    # 🆕 centroid options
    patch_centroid: Optional[Tuple[float, float]] = None,
    include_centroid: bool = False,
    coordinate_format: str = "grid"  # "grid" | "percent"
):
    """
    Generate a natural-language description of the region of interest.

    *patch_centroid* – (row, col) floats in grid coordinates.
    If *include_centroid* is True and *patch_centroid* is provided, a sentence indicating
    the centroid location will be added to the prompt. If *coordinate_format* == "percent",
    the centroid will be expressed as a percentage of the grid's height and width; otherwise
    integer grid coordinates are used.
    """

    # ------------------- derive variable parts -------------------
    size_ratio = patch_size / float(grid_size * grid_size)
    qual_label = label_size(size_ratio)
    display_label = maybe_use_size_synonym(qual_label, prob=0.4)
    region_word = maybe_use_vocab("region", environment_terms, prob=0.5)


    tgt_snip = "the flag"

    opening = "You are leading a team of autonomous robots tasked with exploring and finding " + tgt_snip + "."

    region_line = "The image below represents a simplified map of the environment."
    if include_color and color_name:
        region_line += f" It highlights a {color_name} {region_word} of interest"
    else:
        region_line += f" It highlights a {region_word} of interest"
    region_line += f" — this {region_word} does not show the targets directly but suggests where they are likely located."

    size_line = f"The {region_word} covers ≈ {size_ratio*100:.1f}% of the map, i.e. it is {display_label}."

    # 🆕 Centroid line
    centroid_line = ""
    if include_centroid and patch_centroid is not None:
        row, col = patch_centroid
        if coordinate_format == "percent":
            row_pct = (row / (grid_size - 1)) * 100
            col_pct = (col / (grid_size - 1)) * 100
            centroid_line = (
                f"Its centroid lies roughly at ({row_pct:.1f}%, {col_pct:.1f}%) of the height and width respectively."
            )
        else:
            centroid_line = f"Its centroid lies at grid coordinate ({int(round(row))}, {int(round(col))})."

    # Confidence line
    if include_confidence and confidence is not None:
        conf_text = confidence_txt[confidence]
        confidence_line = (
            f"Provide an assessment of your confidence in the {region_word} as a likely location for the targets. "
            f"You are {conf_text}."
        )
    else:
        confidence_line = ""

    # Mission bullet list
    mission_items = [
        f"Clearly describe the location of the {region_word} in relation to the environment. "
        "Use spatial terms like top-left, south-east, center, or along the lower edge.",
        f"Estimate the size of the {region_word} (you already know it is {display_label}; feel free to repeat or refine).",
    ]
    if include_number:
        mission_items.append("Mention the number of targets.")
    if include_color:
        mission_items.append("Mention the color of the region.")
    if include_centroid and patch_centroid is not None:
        mission_items.append("Reference the centroid position you have been given.")
    if include_confidence and confidence is not None:
        mission_items.append("Provide a confidence assessment.")

    mission_block = "- " + "\n- ".join(mission_items)

    # Assemble full prompt
    prompt_parts = [
        opening,
        region_line,
        "• " + size_line,
    ]
    if centroid_line:
        prompt_parts.append("• " + centroid_line)

    prompt_parts += [
        "",
        "Your mission:",
        mission_block,
        "",
        random.choice(tone_styles),
        "",
        "Do not use any formatting such as bold or italics. Write your response as plain text sentences only. Be Brief",
    ]

    if confidence_line:
        prompt_parts.append("\n" + confidence_line)

    prompt = "\n\n".join(prompt_parts)

    # --------------- call the LLM / vision model ---------------
    img = Image.open(image_buf)
    response = genai_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[prompt, img]
    )
    return response.text

# (describe_image_with_timeout and the rest remain unchanged)

def describe_image_with_timeout(image_buf, timeout=10, **describe_kwargs):
    """Thin wrapper that enforces a time‑out around `describe_image()`."""
    image_buf.seek(0)
    buf_copy = BytesIO(image_buf.read())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(describe_image, buf_copy, **describe_kwargs)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            print("⚠️ Gemini took too long, skipping.")
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")
        return None

# ========================== Main Generation Loop ==========================

def target_and_color():
    output_file = "sequential_tasks/sentences/gemini_patch_dataset_exploration_scale.jsonl"
    json_output_file = "sequential_tasks/sentences/gemini_patch_dataset_exploration_scale.json"
    start_index = 0
    buffer = []

    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            existing_lines = f.readlines()
            start_index = len(existing_lines)
            print(f"📂 Resuming from index {start_index}")

    with open(output_file, "a") as f:
        
        include_color = False
        include_number = False
        include_confidence = False
        include_centroid = False
        for i in tqdm(range(start_index, NUM_DATA_TARGETS), desc="Generating Data"):
            grid, _, _, color_info, patch_sizes = generate_grid_and_patches()
            image_buf = plot_grid_with_rgb(grid, color_info[0]["rgb"])
            patch_cells = patch_sizes[0]  # number of cells in the patch

            # 🆕 Centroid calculation
            centroid = compute_patch_centroid(grid, patch_value=1)  # (row, col) or None

            description = describe_image_with_timeout(
                image_buf,
                grid_size=grid_size,
                patch_size=patch_cells,
                color_name=color_info[0]["name"],
                num_targets=color_info[0]["num_targets"],
                confidence=color_info[0]["confidence"],
                include_color=include_color,
                include_number=include_number,
                include_confidence=include_confidence,
                include_centroid=include_centroid,
                patch_centroid=centroid,        
                coordinate_format="percent"      
            )

            if description is None:
                continue

            time.sleep(API_DELAY_SECONDS)

            entry = {
                "grid": grid.flatten().tolist(),
                "response": description,
            }
            
            if include_color:
                entry["color"] = color_info[0]["index"]
                
            if include_number:
                entry["num_targets"] = color_info[0]["num_targets"]
            
            if include_confidence:
                entry["confidence"] = color_info[0]["confidence"]
            
        
            buffer.append(entry)

            if len(buffer) >= BATCH_SIZE:
                for b in buffer:
                    f.write(json.dumps(b) + "\n")
                buffer.clear()

        for b in buffer:
            f.write(json.dumps(b) + "\n")

    # Save full dataset as JSON for readability
    with open(output_file) as f:
        data = [json.loads(line) for line in f]
    with open(json_output_file, "w") as f:
        json.dump(data, f, indent=2)
        
if __name__ == "__main__":
    target_and_color()
