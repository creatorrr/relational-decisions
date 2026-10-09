"""Audit archived dev results and compare paired worlds; never opens heldout."""

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "benchmarks/nl_logic_v1"))
from logic import solve
from score import LABELS, evaluate, read_jsonl


def read(path):
    return json.loads(path.read_text())


def main():
    inputs = read_jsonl(ROOT / "benchmarks/nl_logic_v1/data/dev.inputs.jsonl")
    gold = read_jsonl(ROOT / "benchmarks/nl_logic_v1/data/dev.gold.jsonl")
    results, predictions, metadata, resources = {}, {}, {}, {}
    for model in ("340m", "1b"):
        prefix = f"decide-{model}-dev."
        pred = read_jsonl(ARTIFACTS / (prefix + "predictions.jsonl"))
        metrics = read(ARTIFACTS / (prefix + "metrics.json"))
        assert evaluate(inputs, gold, pred) == metrics
        queries = 0
        for pid, row in pred.items():
            inp, facts = inputs[pid], []
            for candidate in inp["candidates"]:
                label = row["assessments"][candidate["id"]]
                if label in ("supported", "both"):
                    facts.append(candidate["atom"])
                if label in ("refuted", "both"):
                    facts.append(candidate["negative_atom"])
            reference = solve(inp["program"], facts)["query_probabilities"]
            assert {q: Fraction(p) for q, p in reference.items()} == {
                q: Fraction(p) for q, p in row["query_probabilities"].items()
            }, pid
            queries += len(reference)
        counts = Counter(
            v for row in pred.values() for v in row["assessments"].values()
        )
        confusion = metrics["overall"]["confusion_matrix"]
        results[model] = {
            "overall": metrics["overall"],
            "predicted_labels": {label: counts[label] for label in LABELS},
            "recall_by_label": {
                label: confusion[label][label] / sum(confusion[label].values())
                for label in LABELS
            },
            "paraphrase_assessment_agreement": metrics[
                "paraphrase_assessment_agreement"
            ],
            "paraphrase_query_probability_mean_difference": metrics[
                "paraphrase_query_probability_mean_difference"
            ],
            "verified_reference_programs": len(pred),
            "verified_reference_queries": queries,
            "rescored_metrics_match": True,
        }
        predictions[model] = pred
        metadata[model] = read(ARTIFACTS / (prefix + "metadata.json"))
        resources[model] = read(ARTIFACTS / (prefix + "resources.json"))
        assert metadata[model]["status"] == resources[model]["status"] == "complete"
        assert metadata[model]["backend"]["prompt_version"] == "binary-reports-v2"
    for key in ("config", "input_sha256", "complete_assessments"):
        assert metadata["340m"][key] == metadata["1b"][key]
    assert {
        k: v
        for k, v in metadata["340m"]["backend"].items()
        if k not in ("model", "revision")
    } == {
        k: v
        for k, v in metadata["1b"]["backend"].items()
        if k not in ("model", "revision")
    }
    for key in ("runtime_source_sha256", "protocol_sha256", "selection_sha256"):
        assert resources["340m"][key] == resources["1b"][key]
    assert (
        resources["1b"]["protocol_sha256"]
        == hashlib.sha256((ARTIFACTS / "protocol.json").read_bytes()).hexdigest()
    )
    assert (
        resources["1b"]["selection_sha256"]
        == hashlib.sha256(
            (ARTIFACTS / "train-screen/selection.json").read_bytes()
        ).hexdigest()
    )

    interrupted_path = ARTIFACTS / "decide-340m-dev.interrupted.predictions.jsonl"
    original = read_jsonl(interrupted_path)
    assert len(original) == resources["340m"]["original_complete_programs"] == 39
    assert (
        hashlib.sha256(interrupted_path.read_bytes()).hexdigest()
        == resources["340m"]["original_predictions_sha256"]
    )
    for pid, row in original.items():
        assert predictions["340m"][pid] == row, pid

    # A pair of text renderings is one resampling unit, not two independent worlds.
    groups = defaultdict(list)
    for pid, inp in inputs.items():
        groups[inp["group_id"]].append(pid)
    deltas = []
    for group in sorted(groups):
        differences = []
        assert len(groups[group]) == 2
        for pid in groups[group]:
            for cid, truth in gold[pid]["assessments"].items():
                differences.append(
                    int(predictions["1b"][pid]["assessments"][cid] == truth)
                    - int(predictions["340m"][pid]["assessments"][cid] == truth)
                )
        deltas.append(sum(differences) / len(differences))
    rng = random.Random(20261009)
    draws = sorted(
        sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(5000)
    )

    def percentile(p):
        index = (len(draws) - 1) * p
        lower = int(index)
        return draws[lower] + (draws[min(lower + 1, len(draws) - 1)] - draws[lower]) * (
            index - lower
        )

    comparison = {
        "models": results,
        "matched_model_settings_except_checkpoint": True,
        "matched_evaluated_source_hashes": True,
        "resumed_340m_predictions_match_original_programs": len(original),
        "accuracy_difference_1b_minus_340m": sum(deltas) / len(deltas),
        "exploratory_paired_world_bootstrap": {
            "worlds": len(deltas),
            "draws": 5000,
            "seed": 20261009,
            "percentile_95_interval": [percentile(0.025), percentile(0.975)],
            "interpretation": "Conditional development comparison of a prompt selected on 340M train; not a heldout result or an estimate of each model's best prompt.",
        },
        "timing_note": "340M was interrupted by an environment restart and resumed from the exact-input cache; its metadata elapsed time covers replay and the remaining inference, not a fresh full run.",
        "heldout_contents_inspected_or_evaluated": False,
    }
    (ARTIFACTS / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
