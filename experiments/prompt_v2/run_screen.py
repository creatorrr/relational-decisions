"""Evaluate a frozen train panel, loading one model for all prompt variants.

Run from the repository root. This script never opens dev or heldout data.
"""

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

from relational_decisions import Candidate
from relational_decisions.decisions import best_label, normalize
from relational_decisions.gliner import GLiNERBackend

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/nl_logic_v1"))
from score import summarize


def read_rows(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("protocol.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    data = ROOT / "benchmarks/nl_logic_v1/data"
    assert (
        hashlib.sha256((data / "train.inputs.jsonl").read_bytes()).hexdigest()
        == protocol["train_input_sha256"]
    )
    inputs = read_rows(data / "train.inputs.jsonl")
    # Gold is used only after each prompt's model predictions have been recorded.
    gold = read_rows(data / "train.gold.jsonl")
    model = protocol["selection_model"]
    backend = GLiNERBackend(model["model"], model["revision"], threads=4)
    base_identity = dict(backend.identity)
    all_metrics = {}
    identities = {}
    start = time.perf_counter()
    for prompt in protocol["prompts"]:
        # Each assess call builds and compiles a fresh schema. There is no cache
        # in this screen; changing this field changes the actual model request.
        backend.set_prompt(prompt)
        identities[prompt] = dict(backend.identity)
        predictions = []
        prompt_start = time.perf_counter()
        with (args.output / f"{prompt}.predictions.jsonl").open("w") as file:
            for index, pid in enumerate(protocol["train_ids"], 1):
                inp = inputs[pid]
                candidates = sorted(
                    (Candidate.from_dict(c) for c in inp["candidates"]),
                    key=lambda c: c.id,
                )
                scores = {}
                batch_ids = []
                for offset in range(0, len(candidates), protocol["batch_size"]):
                    batch = tuple(candidates[offset : offset + protocol["batch_size"]])
                    scores.update(backend.assess(inp["world_text"], batch))
                    batch_ids.append([c.id for c in batch])
                row = {
                    "id": pid,
                    "batches": batch_ids,
                    "distributions": scores,
                    "assessments": {
                        cid: best_label(normalize(s)) for cid, s in scores.items()
                    },
                }
                predictions.append(row)
                file.write(json.dumps(row) + "\n")
                file.flush()
                print(f"{prompt} {index}/{len(protocol['train_ids'])}", flush=True)
        records = [
            {
                "labels": [
                    (truth, r["assessments"][cid])
                    for cid, truth in gold[r["id"]]["assessments"].items()
                ],
                "probability_errors": [],
            }
            for r in predictions
        ]
        metrics = summarize(records)
        metrics["elapsed_seconds"] = time.perf_counter() - prompt_start
        all_metrics[prompt] = metrics
        (args.output / "metrics.json").write_text(
            json.dumps(all_metrics, indent=2) + "\n"
        )
        print(prompt, json.dumps(metrics), flush=True)
    # Stable protocol order breaks exact ties after macro-F1 and accuracy.
    selected = max(
        protocol["prompts"],
        key=lambda p: (
            all_metrics[p]["assessment_macro_f1"],
            all_metrics[p]["assessment_accuracy"],
        ),
    )
    result = {
        "selected_prompt": selected,
        "selection_rule": protocol["selection_rule"],
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "elapsed_seconds": time.perf_counter() - start,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "runtime_identity": base_identity,
        "prompt_identities": identities,
        "status": "complete",
    }
    (args.output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
