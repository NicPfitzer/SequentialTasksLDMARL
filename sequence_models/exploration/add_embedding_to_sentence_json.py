import json
from sentence_transformers import SentenceTransformer

EXPLORE = 0
NAVIGATE = 1
DEFEND = 2
IDLE = 3

input_file_dict   = {
    EXPLORE: "sentences/exploration_sentences.json",
    NAVIGATE: "sentences/navigate_to_base.json",
    DEFEND: "sentences/flag_defense_sentences.json",
    IDLE: "sentences/idle_sentences.json"
}

output_file_dict = {
    EXPLORE: "sequence_models/data/language_data_complete_exploration.json",
    NAVIGATE: "sequence_models/data/language_data_complete_navigate_base.json",
    DEFEND: "sequence_models/data/language_data_complete_flag_defense.json",
    IDLE: "sequence_models/data/language_data_complete_idle.json"
}

# ── Config ─────────────────────────────────────────────────────────────
GRID_LENGTH  = 100

# ── Recursive collector ────────────────────────────────────────────────
def collect(obj, sentences, owners, seen_ids):
    """
    Depth-first walk of dicts *and* lists.
    • Push every 'response' string into `sentences`
    • Remember its owning dict in `owners`
    • Normalise *any* 'grid' list to ints
    """
    if isinstance(obj, dict):
        # Fix any grid list we stumble upon
        one_hot = [0] * len(input_file_dict)
        one_hot[TYPE] = 1
        
        if "grid" in obj and isinstance(obj["grid"], list):
            obj["grid"] = [int(x) for x in obj["grid"]] + one_hot
        else:
            one_hot = [0] * len(input_file_dict)
            one_hot[TYPE] = 1
            obj["grid"] = [0] * GRID_LENGTH + one_hot

        # Capture this dict if it directly owns a response
        if "response" in obj and isinstance(obj["response"], str):
            if id(obj) not in seen_ids:            # avoid dup if same dict reached twice
                sentences.append(obj["response"])
                owners.append(obj)
                seen_ids.add(id(obj))

        # Recurse into *all* values
        for v in obj.values():
            collect(v, sentences, owners, seen_ids)

    elif isinstance(obj, list):
        for item in obj:
            collect(item, sentences, owners, seen_ids)

# ── Main ───────────────────────────────────────────────────────────────
def pretrained_llm():
    llm = SentenceTransformer("thenlper/gte-large")
    #llm = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    # print current working directory
    with open(input_file, "r") as f:
        data = json.load(f)                       # list of records

    sentences, owners = [], []
    if isinstance(data, list):
        for record in data:
            collect(record, sentences, owners, set())
    else:
        for record in data.values():
            collect(record, sentences, owners, set())

    if not sentences:                             # nothing to embed → bail early
        print("No responses found; nothing written.")
        return

    # Encode
    embeddings = llm.encode(
        sentences,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,                    # SentenceTransformer supports this
    )

    # Attach
    for owner, emb in zip(owners, embeddings):
        owner["embedding"] = emb.tolist()

    # Write ND-JSON (overwrite, not append)
    with open(output_file, "w") as out:
        json.dump(data, out, indent=2)

    print(f"{len(embeddings)} embeddings saved → {output_file}")

# ── CLI ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Loop through all types
    for t in input_file_dict:
        TYPE = t
        input_file = input_file_dict[TYPE]
        output_file = output_file_dict[TYPE]
        print(f"Processing {input_file} → {output_file}")
        pretrained_llm()