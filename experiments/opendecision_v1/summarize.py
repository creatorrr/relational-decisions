"""Rescore frozen dev predictions, check exact inference, compare paired worlds."""

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


def main():
    inputs_path = ROOT / "benchmarks/nl_logic_v1/data/dev.inputs.jsonl"
    inputs = read_jsonl(inputs_path)
    gold = read_jsonl(inputs_path.with_name("dev.gold.jsonl"))
    predictions, results, metas = {}, {}, {}
    prefixes = {
        "opendecision": HERE / "opendecision-dev",
        "gliner-340m": HERE.parent / "prompt_v2/decide-340m-dev",
        "gliner-1b": HERE.parent / "prompt_v2/decide-1b-dev",
    }
    for name, prefix in prefixes.items():
        pred = read_jsonl(Path(str(prefix) + ".predictions.jsonl"))
        metrics = read(Path(str(prefix) + ".metrics.json"))
        meta = read(Path(str(prefix) + ".metadata.json"))
        assert evaluate(inputs, gold, pred) == metrics
        assert (
            meta["input_sha256"] == hashlib.sha256(inputs_path.read_bytes()).hexdigest()
        )
        assert meta["status"] == "complete"
        assert meta["backend"]["prompt_version"] == "binary-reports-v2"
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
            v for row in pred.values() for v in row["assessments"].values()
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
        predictions[name], metas[name] = pred, meta
    resources = read(HERE / "opendecision-dev.resources.json")
    protocol = read(HERE / "protocol.json")
    assert resources["status"] == "complete"
    assert (
        resources["protocol_sha256"]
        == hashlib.sha256((HERE / "protocol.json").read_bytes()).hexdigest()
    )
    assert resources["selection_sha256"] == protocol["prior_selection_sha256"]
    assert (
        metas["opendecision"]["backend"]["opendecision_source_commit"]
        == protocol["package_commit"]
    )
    for name, prefix in prefixes.items():
        assert metas[name]["config"] == metas["opendecision"]["config"]
        other_resources = read(Path(str(prefix) + ".resources.json"))
        for filename in (
            "decisions.py",
            "diagram.py",
            "engine.py",
            "prompts.py",
            "terms.py",
        ):
            key = "src/relational_decisions/" + filename
            assert (
                resources["runtime_source_sha256"][key]
                == other_resources["runtime_source_sha256"][key]
            )
        for package in ("torch", "transformers", "tokenizers"):
            assert (
                metas[name]["backend"]["packages"][package]
                == metas["opendecision"]["backend"]["packages"][package]
            )
        for key in (
            "task_templates",
            "world_prefix",
            "score_composition",
            "dtype",
            "device",
            "threads",
        ):
            assert metas[name]["backend"][key] == metas["opendecision"]["backend"][key]
    groups = defaultdict(list)
    for pid, row in inputs.items():
        groups[row["group_id"]].append(pid)
    comparisons = {}
    for baseline in ("gliner-340m", "gliner-1b"):
        deltas = []
        for group in sorted(groups):
            assert len(groups[group]) == 2
            differences = [
                int(predictions["opendecision"][pid]["assessments"][cid] == truth)
                - int(predictions[baseline][pid]["assessments"][cid] == truth)
                for pid in groups[group]
                for cid, truth in gold[pid]["assessments"].items()
            ]
            deltas.append(sum(differences) / len(differences))
        rng = random.Random(20261009)
        draws = sorted(
            sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(5000)
        )

        def percentile(p, draws=draws):
            index = (len(draws) - 1) * p
            lower = int(index)
            return draws[lower] + (
                draws[min(lower + 1, len(draws) - 1)] - draws[lower]
            ) * (index - lower)

        comparisons[baseline] = {
            "accuracy_difference_opendecision_minus_baseline": sum(deltas)
            / len(deltas),
            "exploratory_paired_world_bootstrap_95_percentile_interval": [
                percentile(0.025),
                percentile(0.975),
            ],
            "worlds": len(deltas),
            "draws": 5000,
            "seed": 20261009,
        }
    result = {
        "models": results,
        "comparisons": comparisons,
        "matched_questions_labels_engine_and_input": True,
        "matched_core_engine_source_and_torch_transformers_tokenizers": True,
        "serialization_identical_across_models": False,
        "interpretation": "Frozen GLiNER train-selected prompt transferred to OpenDecision with complete questions in the state; different model architecture and native serialization. Development comparison, not a heldout result or each model's optimized prompt.",
        "heldout_contents_inspected_or_evaluated": False,
    }
    (HERE / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
