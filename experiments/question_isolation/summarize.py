"""Audit all complete model/control runs and estimate paired development deltas."""

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "benchmarks/nl_logic_v1"))
from logic import solve
from score import LABELS, evaluate, read_jsonl


def read(path):
    return json.loads(path.read_text())


def percentile(draws, p):
    index = (len(draws) - 1) * p
    lower = int(index)
    return draws[lower] + (draws[min(lower + 1, len(draws) - 1)] - draws[lower]) * (
        index - lower
    )


def main():
    input_path = ROOT / "benchmarks/nl_logic_v1/data/dev.inputs.jsonl"
    inputs = read_jsonl(input_path)
    gold = read_jsonl(input_path.with_name("dev.gold.jsonl"))
    prefixes = {
        "gliner-340m-shared": HERE.parent / "prompt_v2/decide-340m-dev",
        "gliner-340m-isolated-native": HERE / "gliner-isolated-native-dev",
        "gliner-340m-isolated-state": HERE / "gliner-isolated-state-dev",
        "gliner-1b-shared": HERE.parent / "prompt_v2/decide-1b-dev",
        "opendecision-isolated-state": HERE.parent / "opendecision_v1/opendecision-dev",
        "openjev-isolated-native": HERE.parent / "openjev_v1/openjev-dev",
    }
    predictions, metadata, resources, results = {}, {}, {}, {}
    for name, prefix in prefixes.items():
        pred = read_jsonl(Path(str(prefix) + ".predictions.jsonl"))
        metrics = read(Path(str(prefix) + ".metrics.json"))
        meta = read(Path(str(prefix) + ".metadata.json"))
        res = read(Path(str(prefix) + ".resources.json"))
        assert meta["status"] == res["status"] == "complete"
        assert (
            meta["input_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
        )
        assert evaluate(inputs, gold, pred) == metrics
        queries = 0
        for pid, row in pred.items():
            facts = []
            for candidate in inputs[pid]["candidates"]:
                label = row["assessments"][candidate["id"]]
                if label in ("supported", "both"):
                    facts.append(candidate["atom"])
                if label in ("refuted", "both"):
                    facts.append(candidate["negative_atom"])
            reference = solve(inputs[pid]["program"], facts)["query_probabilities"]
            assert {q: Fraction(p) for q, p in reference.items()} == {
                q: Fraction(p) for q, p in row["query_probabilities"].items()
            }, pid
            queries += len(reference)
        counts = Counter(
            label for row in pred.values() for label in row["assessments"].values()
        )
        results[name] = {
            "overall": metrics["overall"],
            "predicted_labels": {label: counts[label] for label in LABELS},
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
        predictions[name], metadata[name], resources[name] = pred, meta, res
    base = "gliner-340m-shared"
    for name in prefixes:
        assert metadata[name]["config"] == metadata[base]["config"]
        for key in (
            "prompt_version",
            "task_templates",
            "world_prefix",
            "score_composition",
            "dtype",
            "device",
            "threads",
        ):
            assert metadata[name]["backend"][key] == metadata[base]["backend"][key]
        for package in ("torch", "transformers", "tokenizers"):
            assert (
                metadata[name]["backend"]["packages"][package]
                == metadata[base]["backend"]["packages"][package]
            )
        for filename in (
            "decisions.py",
            "diagram.py",
            "engine.py",
            "prompts.py",
            "terms.py",
        ):
            key = "src/relational_decisions/" + filename
            assert (
                resources[name]["runtime_source_sha256"][key]
                == resources[base]["runtime_source_sha256"][key]
            )
    baseline_trace = read_jsonl(Path(str(prefixes[base]) + ".traces.jsonl"))
    for name, layout in (
        ("gliner-340m-isolated-native", "isolated-native"),
        ("gliner-340m-isolated-state", "isolated-question-in-state"),
    ):
        assert metadata[name]["backend"]["model"] == metadata[base]["backend"]["model"]
        assert (
            metadata[name]["backend"]["revision"]
            == metadata[base]["backend"]["revision"]
        )
        assert metadata[name]["backend"]["question_layout"] == layout
        assert (
            resources[name]["protocol_sha256"]
            == hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()
        )
        trace = read_jsonl(Path(str(prefixes[name]) + ".traces.jsonl"))
        for pid, row in trace.items():
            assert [r["candidate_ids"] for r in row["replayed_model_requests"]] == [
                r["candidate_ids"] for r in baseline_trace[pid]["trace"]
            ]
            assert not any(r["cache_hit"] for r in row["replayed_model_requests"])
    groups = defaultdict(list)
    for pid, row in inputs.items():
        groups[row["group_id"]].append(pid)
    comparisons = {}
    pairs = [
        ("gliner-340m-isolated-native", base),
        ("gliner-340m-isolated-state", "gliner-340m-isolated-native"),
        ("opendecision-isolated-state", "gliner-340m-isolated-state"),
        ("openjev-isolated-native", "gliner-340m-isolated-native"),
    ]
    for treatment, control in pairs:
        differences = []
        for group in sorted(groups):
            assert len(groups[group]) == 2
            outcomes = [
                int(predictions[treatment][pid]["assessments"][cid] == truth)
                - int(predictions[control][pid]["assessments"][cid] == truth)
                for pid in groups[group]
                for cid, truth in gold[pid]["assessments"].items()
            ]
            differences.append(sum(outcomes) / len(outcomes))
        rng = random.Random(20261009)
        draws = sorted(
            sum(rng.choices(differences, k=len(differences))) / len(differences)
            for _ in range(5000)
        )
        comparisons[treatment + " minus " + control] = {
            "accuracy_difference": sum(differences) / len(differences),
            "paired_world_bootstrap_95_percentile_interval": [
                percentile(draws, 0.025),
                percentile(draws, 0.975),
            ],
            "worlds": len(differences),
            "draws": 5000,
            "seed": 20261009,
        }
    truth_counts = Counter(
        t for row in gold.values() for t in row["assessments"].values()
    )
    result = {
        "models": results,
        "comparisons": comparisons,
        "majority_label_accuracy": max(truth_counts.values())
        / sum(truth_counts.values()),
        "matched_core_engine_source_questions_labels_and_numerical_packages": True,
        "isolation_controls_replay_exact_baseline_request_groups": True,
        "interpretation": "Exploratory dev interventions, proposed after interim OpenDecision observations; no heldout claim. The GLiNER native isolation control changes only which questions share an input sequence, while the state condition changes question placement and task name. Cross-model comparisons still differ in architecture and native serialization; OpenDecision also isolates answer options.",
        "heldout_contents_inspected_or_evaluated": False,
    }
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
