"""Create v1 once. Refuse overwriting frozen data. No model is involved."""
import collections
import gzip
import hashlib
import io
import json
import random
import tarfile
from pathlib import Path

from logic import solve, to_problog

ROOT = Path(__file__).resolve().parent
VERSION = "nl-logic-v1"
SEED = "nl-logic-v1-frozen-2026-10-09"
SPLITS = {"train": range(6), "dev": range(6, 8), "heldout": range(8, 10)}
FAMILIES = [
    "conjunction", "disjunction", "shared_diamond", "repeated_choice",
    "exclusive_choice", "staged_chain", "cyclic_reachability",
    "left_recursion", "binding_join", "contradictory_reports",
    "missing_vs_negative", "conditioning",
]
STATES = ["supported", "refuted", "both", "unknown"]
THEMES = {
    "rooms": {
        "noun": "Room",
        "needs_attention": ("{x} needs maintenance", "{x} does not need maintenance"),
        "ready": ("{x} is available for occupancy", "{x} is unavailable for occupancy"),
        "requested": ("A room change has been requested for {x}", "No room change has been requested for {x}"),
    },
    "deliveries": {
        "noun": "Package",
        "needs_attention": ("{x} needs inspection", "{x} does not need inspection"),
        "ready": ("{x} is ready for dispatch", "{x} is not ready for dispatch"),
        "requested": ("Priority handling has been requested for {x}", "No priority handling has been requested for {x}"),
    },
    "gardens": {
        "noun": "Garden",
        "needs_attention": ("{x} needs watering", "{x} does not need watering"),
        "ready": ("{x} is accessible", "{x} is inaccessible"),
        "requested": ("A caretaker visit has been requested for {x}", "No caretaker visit has been requested for {x}"),
    },
}
SURFACES = {
    "train": ["{s}.", "The log states: {s}."],
    "dev": ["According to the current record: {s}.", "Recorded status: {s}."],
    "heldout": ["An entry explicitly reports: {s}.", "The status register contains this assertion: {s}."],
}
PARAPHRASES = {
    "dev": {
        "rooms": {
            "needs_attention": ("Maintenance work is required in {x}", "Maintenance work is not required in {x}"),
            "ready": ("{x} can accommodate an occupant", "{x} cannot accommodate an occupant"),
            "requested": ("There is a request to change rooms for {x}", "There is no request to change rooms for {x}"),
        },
        "deliveries": {
            "needs_attention": ("An inspection is required for {x}", "An inspection is not required for {x}"),
            "ready": ("{x} can now be dispatched", "{x} cannot yet be dispatched"),
            "requested": ("There is a request to prioritize {x}", "There is no request to prioritize {x}"),
        },
        "gardens": {
            "needs_attention": ("{x} requires watering", "{x} does not require watering"),
            "ready": ("Access to {x} is possible", "Access to {x} is impossible"),
            "requested": ("There is a request for a caretaker to visit {x}", "There is no request for a caretaker to visit {x}"),
        },
    },
    "heldout": {
        "rooms": {
            "needs_attention": ("{x} requires maintenance attention", "{x} requires no maintenance attention"),
            "ready": ("{x} is open for occupancy", "{x} is closed to occupancy"),
            "requested": ("A request for a room move exists for {x}", "A request for a room move does not exist for {x}"),
        },
        "deliveries": {
            "needs_attention": ("{x} must undergo inspection", "{x} need not undergo inspection"),
            "ready": ("{x} is in a dispatch-ready state", "{x} is not in a dispatch-ready state"),
            "requested": ("A priority-service request exists for {x}", "A priority-service request does not exist for {x}"),
        },
        "gardens": {
            "needs_attention": ("Watering is needed in {x}", "Watering is not needed in {x}"),
            "ready": ("{x} can be accessed", "{x} cannot be accessed"),
            "requested": ("A request for caretaker attendance exists for {x}", "A request for caretaker attendance does not exist for {x}"),
        },
    },
}
PREFIXES = {"train": "Cedar", "dev": "Maple", "heldout": "Willow"}


def rng_for(*parts):
    return random.Random(int.from_bytes(hashlib.sha256((SEED + "/" + "/".join(map(str, parts))).encode()).digest(), "big"))


def rule(head, *body):
    return {"head": head, "body": list(body)}


def bernoulli(name, atom, p):
    return {"id": name, "outcomes": [
        {"p": str(p), "facts": [atom]}, {"p": str(1-p), "facts": []},
    ]}


def skeleton(family, entities, rng):
    from fractions import Fraction
    a, b, c, d = entities
    program = {"facts": [["entity", e] for e in entities], "rules": [], "choices": [], "evidence": [], "queries": []}
    ps = [Fraction(rng.choice([1, 2, 3, 4, 6, 7, 8, 9]), 10) for _ in range(6)]
    for i in range(3):
        program["choices"].append(bernoulli(f"g{i}", ["gate", f"g{i}"], ps[i]))
    r = program["rules"]
    def query(name, atom):
        program["queries"].append({"id": name, "atom": atom})
    if family == "conjunction":
        r += [rule(["act", "?x"], ["needs_attention", "?x"], ["ready", "?x"], ["gate", "g0"], ["gate", "g1"])]
    elif family == "disjunction":
        r += [rule(["act", "?x"], ["needs_attention", "?x"], ["gate", "g0"]),
              rule(["act", "?x"], ["requested", "?x"], ["gate", "g1"])]
    elif family == "shared_diamond":
        r += [rule(["left", "?x"], ["needs_attention", "?x"], ["gate", "g0"], ["gate", "g1"]),
              rule(["right", "?x"], ["requested", "?x"], ["gate", "g0"], ["gate", "g2"]),
              rule(["act", "?x"], ["left", "?x"]), rule(["act", "?x"], ["right", "?x"])]
        query("shared_left", ["left", a]); query("shared_right", ["right", a])
    elif family == "repeated_choice":
        r += [rule(["act", "?x"], ["needs_attention", "?x"], ["gate", "g0"], ["gate", "g0"]),
              rule(["single_use", "?x"], ["needs_attention", "?x"], ["gate", "g0"])]
        query("single_use_a", ["single_use", a])
    elif family == "exclusive_choice":
        weights = [rng.randint(1, 8) for _ in range(3)]
        total = sum(weights)
        program["choices"].append({"id": "mode", "outcomes": [
            {"p": str(Fraction(w,total)), "facts": [["mode", label]]}
            for w,label in zip(weights,["fast", "standard", "offline"])
        ]})
        r += [rule(["act", "?x"], ["requested", "?x"], ["mode", "fast"]),
              rule(["act", "?x"], ["requested", "?x"], ["mode", "standard"]),
              rule(["impossible", "?x"], ["entity", "?x"], ["mode", "fast"], ["mode", "offline"])]
        query("exclusive_impossible", ["impossible", a])
    elif family == "staged_chain":
        r += [rule(["stage1", "?x"], ["needs_attention", "?x"], ["gate", "g0"]),
              rule(["stage2", "?x"], ["stage1", "?x"], ["requested", "?x"], ["gate", "g1"]),
              rule(["act", "?x"], ["stage2", "?x"], ["ready", "?x"], ["gate", "g2"])]
        query("stage1_a", ["stage1", a]); query("stage2_a", ["stage2", a])
    elif family in ("cyclic_reachability", "left_recursion"):
        edges = [(a,b), (b,c), (c,a), (c,d)]
        for i,(x,y) in enumerate(edges):
            r.append(rule(["edge",x,y], ["gate", f"g{i%3}"]))
        if family == "left_recursion":
            r += [rule(["reach","?x","?y"], ["reach","?x","?z"], ["edge","?z","?y"]),
                  rule(["reach","?x","?y"], ["edge","?x","?y"])]
        else:
            r += [rule(["reach","?x","?y"], ["edge","?x","?y"]),
                  rule(["reach","?x","?y"], ["edge","?x","?z"], ["reach","?z","?y"])]
        r += [rule(["act","?x"], ["needs_attention","?x"], ["reach","?x","?y"], ["ready","?y"])]
        query("reach_a_d", ["reach",a,d]); query("reach_a_a", ["reach",a,a]); query("reach_d_a", ["reach",d,a])
    elif family == "binding_join":
        program["facts"] += [["route",a,b], ["route",a,c], ["route",c,d]]
        r += [rule(["transfer","?x","?y"], ["requested","?x"], ["route","?x","?y"], ["ready","?y"], ["gate","g0"]),
              rule(["act","?x"], ["transfer","?x","?y"])]
        query("transfer_a_b", ["transfer",a,b]); query("transfer_b_a", ["transfer",b,a])
    elif family == "contradictory_reports":
        r += [rule(["disputed","?x"], ["needs_attention","?x"], ["not_needs_attention","?x"]),
              rule(["act","?x"], ["disputed","?x"], ["gate","g0"]),
              rule(["unrelated","?x"], ["ready","?x"], ["gate","g1"])]
        query("disputed_a", ["disputed",a]); query("unrelated_b", ["unrelated",b])
    elif family == "missing_vs_negative":
        r += [rule(["blocked","?x"], ["not_ready","?x"]),
              rule(["act","?x"], ["ready","?x"], ["gate","g0"])]
        query("blocked_a", ["blocked",a]); query("blocked_b", ["blocked",b])
    elif family == "conditioning":
        r += [rule(["alarm"], ["gate","g0"]), rule(["alarm"], ["gate","g1"]),
              rule(["act","?x"], ["needs_attention","?x"], ["gate","g0"])]
        program["evidence"] = [{"atom":["alarm"], "value":True}]
        query("cause_given_alarm", ["gate","g0"]); query("alarm", ["alarm"])
    for i,e in enumerate(entities):
        query(f"act_{i}", ["act",e])
    return program


def build_group(split, family, index, family_index):
    rng = rng_for(family, index)
    group_id = hashlib.sha256(f"{VERSION}/{family}/{index}".encode()).hexdigest()[:16]
    theme_id = list(THEMES)[(family_index + index) % len(THEMES)]
    theme = THEMES[theme_id]
    entity_ids = [f"{PREFIXES[split].lower()}_{family_index:02d}_{index}_{i}" for i in range(4)]
    display = {e:f"{theme['noun']} {PREFIXES[split]}-{family_index+1:02d}{index}{i}" for i,e in enumerate(entity_ids)}
    program = skeleton(family, entity_ids, rng)
    labels = {}
    candidates = []
    oracle_facts = []
    reports = []
    # Start balanced, then force the diagnostic cases for selected families.
    assignments = STATES * 3
    rng.shuffle(assignments)
    keys = [(pred,e) for e in entity_ids for pred in ("needs_attention","ready","requested")]
    if family in ("conjunction", "shared_diamond", "repeated_choice", "staged_chain", "conditioning"):
        for key in [("needs_attention",entity_ids[0]), ("ready",entity_ids[0]), ("requested",entity_ids[0])]:
            assignments[keys.index(key)] = "supported"
    if family == "exclusive_choice":
        assignments[keys.index(("requested",entity_ids[0]))] = "supported"
    if family == "contradictory_reports":
        assignments[keys.index(("needs_attention",entity_ids[0]))] = "both"
        assignments[keys.index(("ready",entity_ids[1]))] = "unknown"
    if family == "missing_vs_negative":
        assignments[keys.index(("ready",entity_ids[0]))] = "unknown"
        assignments[keys.index(("ready",entity_ids[1]))] = "refuted"
    for n,((pred,e), state) in enumerate(zip(keys,assignments)):
        cid = f"p{n:02d}"
        positive = theme[pred][0].format(x=display[e])
        negative = theme[pred][1].format(x=display[e])
        candidates.append({"id":cid, "atom":[pred,e], "negative_atom":["not_"+pred,e], "proposition":positive, "negative_proposition":negative})
        labels[cid] = state
        rendering = PARAPHRASES.get(split, {}).get(theme_id, {}).get(pred, theme[pred])
        if state in ("supported","both"):
            reports.append(rendering[0].format(x=display[e])); oracle_facts.append([pred,e])
        if state in ("refuted","both"):
            reports.append(rendering[1].format(x=display[e])); oracle_facts.append(["not_"+pred,e])
    answers = solve(program, oracle_facts)
    semantic_fingerprint = hashlib.sha256(json.dumps({
        "family":family, "states":assignments,
        "probabilities":[[o["p"] for o in c["outcomes"]] for c in program["choices"]],
    },sort_keys=True).encode()).hexdigest()
    inputs, gold = [], []
    for variant in range(2):
        # Same semantics, independent sentence order and surface realization.
        surface_rng = rng_for(family,index,variant,"surface")
        ordered_reports = list(reports)
        surface_rng.shuffle(ordered_reports)
        sentence = SURFACES[split][variant]
        world_text = "\n".join(sentence.format(s=s) for s in ordered_reports)
        pid = group_id + f"-v{variant}"
        inp = {
            "id":pid, "group_id":group_id, "family":family, "domain":theme_id,
            "world_text":world_text,
            "entities":[{"id":e,"name":display[e]} for e in entity_ids],
            "predicates":[{"name":p,"arity":1,"definition":theme[p][0].format(x="{entity}")}
                          for p in ("needs_attention","ready","requested")],
            "candidates":candidates, "program":program,
        }
        target = {"id":pid, "group_id":group_id, "assessments":labels,
                  "oracle_facts":oracle_facts, **answers}
        inputs.append(inp); gold.append(target)
    return inputs, gold, semantic_fingerprint


def jsonl(rows):
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def main():
    if (ROOT / "manifest.json").exists() or (ROOT / "data").exists() or (ROOT / "sealed").exists():
        raise SystemExit("Frozen outputs already exist. Create a new benchmark version instead of overwriting.")
    partitions = {s:([],[]) for s in SPLITS}
    signatures = set()
    group_sets = {}
    for split, indices in SPLITS.items():
        group_sets[split] = set()
        for fi,family in enumerate(FAMILIES):
            for index in indices:
                inputs,gold,fingerprint = build_group(split,family,index,fi)
                assert fingerprint not in signatures, "Duplicate semantics modulo entity renaming"
                signatures.add(fingerprint)
                partitions[split][0].extend(inputs); partitions[split][1].extend(gold)
                group_sets[split].add(inputs[0]["group_id"])
        rng_for(split,"row_order").shuffle(partitions[split][0])
        rng_for(split,"gold_order").shuffle(partitions[split][1])
    assert not any(group_sets[a] & group_sets[b] for a in SPLITS for b in SPLITS if a != b)
    (ROOT / "data").mkdir(); (ROOT / "sealed").mkdir()
    files = {}
    for split in ("train","dev"):
        for kind,rows in zip(("inputs","gold"),partitions[split]):
            path = ROOT / "data" / f"{split}.{kind}.jsonl"
            payload = jsonl(rows); path.write_bytes(payload)
            files[str(path.relative_to(ROOT))] = hashlib.sha256(payload).hexdigest()
    # Keep both test prompts and answers out of the development data directory.
    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        for kind,rows in zip(("inputs","gold"),partitions["heldout"]):
            payload = jsonl(rows)
            info = tarfile.TarInfo(f"heldout.{kind}.jsonl")
            info.size = len(payload); info.mtime = 0; info.mode = 0o400
            archive.addfile(info, io.BytesIO(payload))
    sealed = ROOT / "sealed" / "heldout-v1.tar.gz"
    sealed.write_bytes(gzip.compress(tar_buffer.getvalue(), mtime=0))
    files[str(sealed.relative_to(ROOT))] = hashlib.sha256(sealed.read_bytes()).hexdigest()
    counts = {}
    for split,(inputs,gold) in partitions.items():
        counts[split] = {
            "groups":len(group_sets[split]), "programs":len(inputs),
            "queries":sum(len(i["program"]["queries"]) for i in inputs),
            "assessments":sum(len(i["candidates"]) for i in inputs),
            "families":dict(collections.Counter(i["family"] for i in inputs)),
        }
    manifest = {
        "version":VERSION, "created":"2026-10-09", "seed":SEED,
        "split_unit":"underlying symbolic world; both renderings stay together",
        "split_type":"stratified within-family; split-specific surface templates and entity vocabularies",
        "counts":counts, "sha256":files,
        "ground_truth":"exact rational enumeration of independent categorical choices and least-fixed-point positive Datalog",
        "heldout_policy":"No model predictions, prompt selection, calibration, or error inspection on heldout until final evaluation.",
        "model_evaluations_at_creation":0,
    }
    (ROOT / "manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    # One open development example for easy inspection; never select from heldout.
    example = next(i for i in partitions["dev"][0] if i["family"] == "shared_diamond")
    target = next(g for g in partitions["dev"][1] if g["id"] == example["id"])
    (ROOT / "example.dev.json").write_text(json.dumps({"input":example,"gold":target},indent=2)+"\n")
    (ROOT / "example.dev.pl").write_text(to_problog(example["program"],target["oracle_facts"]))
    print(json.dumps({"counts":counts,"heldout_sha256":files[str(sealed.relative_to(ROOT))]},indent=2))


if __name__ == "__main__":
    main()
