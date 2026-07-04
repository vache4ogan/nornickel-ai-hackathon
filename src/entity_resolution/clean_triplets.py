#!/usr/bin/env python3
"""
Entity resolution and Strict Ontology filtering for graph_triplets.jsonl
"""

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

INPUT = Path("data/processed/graph_triplets.jsonl")
OUTPUT = Path("data/processed/cleaned_graph_triplets.jsonl")
THRESHOLD = 0.85

STOPWORDS = {"процесс", "система", "метод", "способ", "технология", "данные", "результат", "значение", "установка", "работа", "the", "a", "an"}
ALLOWED_SHORTS = {"cu", "fe", "ni", "co", "pd", "pt", "au", "ag", "so2", "h2s", "ph"}

# --- НОВАЯ СТРОГАЯ ОНТОЛОГИЯ ---
ALLOWED_ENTITIES = {"Material", "Process", "Equipment", "Property", "Experiment", "Publication", "Expert", "Facility"}
ALLOWED_RELATIONS = {"uses_material", "operates_at_condition", "produces_output", "described_in", "validated_by", "contradicts"}

def normalize(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r'^[^\w]+|[^\w]+$', '', name)
    return name

def is_garbage(name: str) -> bool:
    norm = normalize(name)
    if len(norm) < 2 and norm not in ALLOWED_SHORTS:
        return True
    if norm in STOPWORDS:
        return True
    if norm.isnumeric():
        return True
    return False

def build_canonical_map(all_names: list[str]) -> dict[str, str]:
    unique_names = sorted(set(all_names), key=len)
    canonical: dict[str, str] = {}
    clusters: list[tuple[str, list[str]]] = []

    for name in unique_names:
        placed = False
        for seed, members in clusters:
            if seed in name and len(name) - len(seed) <= 3:
                members.append(name)
                placed = True
                break
                
            if SequenceMatcher(None, name, seed).ratio() >= THRESHOLD:
                members.append(name)
                placed = True
                break
                
        if not placed:
            clusters.append((name, [name]))

    for seed, members in clusters:
        for m in members:
            canonical[m] = seed
    return canonical

def main():
    if not INPUT.exists():
        print(f"{INPUT} not found — nothing to clean.")
        return

    triplets = []
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    triplets.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    all_valid_ids = []
    for t in triplets:
        for e in t.get("entities", []):
            # Маппинг старых типов в новые, чтобы не перепаршивать
            if e.get("type") == "Metric":
                e["type"] = "Property"
                
            if not is_garbage(e["id"]) and e.get("type") in ALLOWED_ENTITIES:
                all_valid_ids.append(e["id"])

    normalized_ids = [normalize(n) for n in all_valid_ids]
    canon_map = build_canonical_map(normalized_ids)

    renamed = 0
    dropped_entities = 0
    dropped_relations = 0
    
    cleaned_triplets = []
    
    for t in triplets:
        lookup: dict[str, str] = {}
        
        for e in t.get("entities", []):
            old_id = e["id"]
            ent_type = e.get("type")
            
            # Строгая фильтрация по онтологии и мусору
            if is_garbage(old_id) or ent_type not in ALLOWED_ENTITIES:
                dropped_entities += 1
                continue
                
            norm_id = normalize(old_id)
            new_id = canon_map.get(norm_id, norm_id)
            
            if new_id != norm_id:
                renamed += 1
                
            e["id"] = new_id.title() 
            lookup[old_id] = e["id"]
            
        seen = set()
        deduped_entities = []
        for e in t.get("entities", []):
            if e["id"] in lookup.values() and e.get("type") in ALLOWED_ENTITIES:
                key = (e["id"], e["type"])
                if key not in seen:
                    seen.add(key)
                    deduped_entities.append(e)
                    
        t["entities"] = deduped_entities

        valid_relations = []
        for r in t.get("relations", []):
            rel_type = r.get("type")
            
            # Маппинг старых типов связей в новые
            if rel_type == "PRODUCES":
                rel_type = "produces_output"
            elif rel_type == "USES":
                rel_type = "uses_material"
            elif rel_type == "HAS_PARAMETER":
                rel_type = "operates_at_condition"
            
            r["type"] = rel_type
            
            if r["source"] in lookup and r["target"] in lookup and rel_type in ALLOWED_RELATIONS:
                r["source"] = lookup[r["source"]]
                r["target"] = lookup[r["target"]]
                
                if r["source"] != r["target"]:
                    valid_relations.append(r)
                else:
                    dropped_relations += 1
            else:
                dropped_relations += 1
                
        t["relations"] = valid_relations
        
        if t["entities"] or t["relations"]:
            cleaned_triplets.append(t)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for t in cleaned_triplets:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"Entities dropped (garbage or bad ontology): {dropped_entities}")
    print(f"Relations dropped (garbage, self-loops or bad ontology): {dropped_relations}")
    print(f"Entity occurrences renamed: {renamed}")
    print(f"Output written to: {OUTPUT}")


if __name__ == "__main__":
    main()
