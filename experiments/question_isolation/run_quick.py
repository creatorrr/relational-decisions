"""Eight-program diagnostic after the user stopped the exhaustive control sweep."""

import argparse
import hashlib
import json
import resource
import sys
import time
from fractions import Fraction
from pathlib import Path

from isolated import IsolatedGLiNERBackend
from run_dev import RecordedBackend, read_rows

from relational_decisions import Candidate
from relational_decisions.decisions import DecisionCache
from relational_decisions.engine import Engine, EngineConfig
from relational_decisions.openjev import OpenJevBackend

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/nl_logic_v1"))
from logic import solve
from score import summarize


def score_subset(inputs, predictions, gold):
    records = []
    queries = 0
    for pid, inp in inputs.items():
        pred, truth = predictions[pid], gold[pid]
        assert set(pred["assessments"]) == set(truth["assessments"])
        assert set(pred["query_probabilities"]) == set(truth["query_probabilities"])
        facts = []
        for candidate in inp["candidates"]:
            label = pred["assessments"][candidate["id"]]
            if label in ("supported", "both"):
                facts.append(candidate["atom"])
            if label in ("refuted", "both"):
                facts.append(candidate["negative_atom"])
        reference = solve(inp["program"], facts)["query_probabilities"]
        assert {q: Fraction(p) for q, p in reference.items()} == {
            q: Fraction(p) for q, p in pred["query_probabilities"].items()
        }
        queries += len(reference)
        records.append(
            {
                "labels": [
                    (v, pred["assessments"][c]) for c, v in truth["assessments"].items()
                ],
                "probability_errors": [
                    abs(
                        float(Fraction(v))
                        - float(Fraction(pred["query_probabilities"][q]))
                    )
                    for q, v in truth["query_probabilities"].items()
                ],
            }
        )
    return {
        "overall": summarize(records),
        "verified_reference_queries": queries,
        "scope": "diagnostic subset; no paraphrase-pair or heldout claim",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["gliner-state", "openjev"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache_path = args.output.with_name(args.output.name + "-cache")
    for path in (args.output, cache_path):
        if path.exists() and any(path.iterdir()):
            parser.error("Output and cache must be new")
    source = ROOT / "benchmarks/nl_logic_v1/data/dev.inputs.jsonl"
    inputs = dict(list(read_rows(source).items())[:8])
    baseline = read_rows(ROOT / "experiments/prompt_v2/decide-340m-dev.traces.jsonl")
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    resources = {
        "status": "failed",
        "command": sys.argv,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        if args.model == "gliner-state":
            backend = IsolatedGLiNERBackend(
                "fastino/GLiNER2.5-Decide",
                "9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90",
                layout="isolated-question-in-state",
            )
        else:
            backend = OpenJevBackend()
        meta = {
            "status": "running",
            "scope": "first 8 dev input records; diagnostic only",
            "reason": "User requested useful quick learning instead of completing every full sweep",
            "input_ids": list(inputs),
            "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "backend": backend.identity,
            "heldout_opened": False,
        }
        (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        cache = DecisionCache(cache_path)
        with (
            (args.output / "predictions.jsonl").open("w") as preds,
            (args.output / "traces.jsonl").open("w") as traces,
        ):
            for index, (pid, inp) in enumerate(inputs.items(), 1):
                tick = time.perf_counter()
                candidates = {c["id"]: Candidate(**c) for c in inp["candidates"]}
                distributions, requests = {}, []
                for request in baseline[pid]["trace"]:
                    ids = request["candidate_ids"]
                    values, key, hit = cache.evaluate(
                        backend, inp["world_text"], tuple(candidates[c] for c in ids)
                    )
                    distributions.update(values)
                    requests.append(
                        {"candidate_ids": ids, "cache_key": key, "cache_hit": hit}
                    )
                assert set(distributions) == set(candidates)
                result = (
                    Engine(
                        inp["program"],
                        candidates=inp["candidates"],
                        world=inp["world_text"],
                        backend=RecordedBackend(distributions),
                        config=EngineConfig(),
                    )
                    .run(complete_assessments=True)
                    .to_dict()
                )
                preds.write(
                    json.dumps(
                        {
                            "id": pid,
                            "assessments": result["assessments"],
                            "query_probabilities": result["query_probabilities"],
                        }
                    )
                    + "\n"
                )
                traces.write(
                    json.dumps(
                        {"id": pid, **result, "replayed_model_requests": requests}
                    )
                    + "\n"
                )
                preds.flush()
                traces.flush()
                print(
                    f"{index}/8 {inp['family']}: {time.perf_counter() - tick:.2f}s",
                    flush=True,
                )
        metrics = score_subset(
            inputs,
            read_rows(args.output / "predictions.jsonl"),
            read_rows(source.with_name("dev.gold.jsonl")),
        )
        (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        meta["status"] = resources["status"] = "complete"
        (args.output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(metrics, indent=2))
    finally:
        resources.update(
            wall_seconds=time.perf_counter() - start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        (args.output / "resources.json").write_text(
            json.dumps(resources, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
