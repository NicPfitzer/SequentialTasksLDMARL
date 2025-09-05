import json
from sentence_transformers import SentenceTransformer
from scenarios.four_rooms.language import FIND_GOAL, FIND_FIRST_SWITCH, FIND_SECOND_SWITCH, FIND_THIRD_SWITCH

input_file_dict   = {
    FIND_FIRST_SWITCH: "sentences/navigate_to_first_switch.json",
    FIND_SECOND_SWITCH: "sentences/navigate_to_second_switch.json",
    FIND_THIRD_SWITCH: "sentences/navigate_to_third_switch.json",
    FIND_GOAL: "sentences/navigate_to_goal.json",
}

output_file_dict = {
    FIND_FIRST_SWITCH: "sequence_models/data/language_data_complete_first_switch.json",
    FIND_SECOND_SWITCH: "sequence_models/data/language_data_complete_second_switch.json",
    FIND_THIRD_SWITCH: "sequence_models/data/language_data_complete_third_switch.json",
    FIND_GOAL: "sequence_models/data/language_data_complete_goal.json",
}

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
        
        one_hot = [0] * len(input_file_dict)
        one_hot[TYPE] = 1
        obj["subtask_decoder_label"] = one_hot

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