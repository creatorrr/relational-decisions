"""Replay frozen GLiNER request groups, then evaluate their recorded decisions."""

import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from isolated import LAYOUTS, IsolatedGLiNERBackend

from relational_decisions import Candidate
from relational_decisions.decisions import DecisionCache
from relational_decisions.engine import Engine, EngineConfig

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def read_rows(path):
    return {
        r["id"]: r for line in path.read_text().splitlines() if (r := json.loads(line))
    }


class RecordedBackend:
    def __init__(self, distributions):
        self.distributions = distributions

    @property
    def identity(self):
        return {
            "backend": "recorded-model-assessments",
            "source": "baseline request replay",
        }

    def assess(self, world, candidates):
        return {c.id: self.distributions[c.id] for c in candidates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", choices=LAYOUTS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = (HERE / "protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    prefix = ROOT / protocol["baseline"]
    trace_path = Path(str(prefix) + ".traces.jsonl")
    pred_path = Path(str(prefix) + ".predictions.jsonl")
    for path, key in (
        (trace_path, "baseline_traces_sha256"),
        (pred_path, "baseline_predictions_sha256"),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != protocol[key]:
            parser.error("Frozen baseline hash mismatch")
    output = args.output
    cache_path = output.with_name(output.name + "-cache")
    for path in (output, cache_path):
        if path.exists() and any(path.iterdir()):
            parser.error("Output and cache must be new")
    inputs_path = ROOT / "benchmarks/nl_logic_v1/data/dev.inputs.jsonl"
    inputs, baseline = read_rows(inputs_path), read_rows(trace_path)
    if set(inputs) != set(baseline):
        parser.error("Baseline and input IDs differ")
    output.mkdir(parents=True)
    resources = {
        "status": "failed",
        "command": sys.argv,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "runtime_source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((ROOT / "src/relational_decisions").glob("*.py"))
        },
        "experiment_source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(HERE.glob("*.py"))
        },
    }
    start = time.perf_counter()
    try:
        backend = IsolatedGLiNERBackend(
            protocol["model"], protocol["revision"], layout=args.layout
        )
        baseline_meta = json.loads(Path(str(prefix) + ".metadata.json").read_text())
        config = EngineConfig(**baseline_meta["config"])
        meta = {
            "created": datetime.now(UTC).isoformat(),
            "status": "running",
            "split": "dev",
            "backend": backend.identity,
            "config": baseline_meta["config"],
            "input_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
            "programs": len(inputs),
            "complete_assessments": True,
            "model_requests": "Exact replay of baseline trace groups before symbolic evaluation",
            "baseline_traces_sha256": protocol["baseline_traces_sha256"],
        }
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        cache = DecisionCache(cache_path)
        inference_start = time.perf_counter()
        calls = 0
        with (
            (output / "predictions.jsonl").open("w") as preds,
            (output / "traces.jsonl").open("w") as traces,
        ):
            for index, (pid, inp) in enumerate(inputs.items(), 1):
                program_start = time.perf_counter()
                candidates = {c["id"]: Candidate(**c) for c in inp["candidates"]}
                distributions, requests = {}, []
                for request in baseline[pid]["trace"]:
                    ids = request["candidate_ids"]
                    if set(ids) & set(distributions):
                        raise ValueError("Baseline assesses a candidate more than once")
                    tick = time.perf_counter()
                    values, key, hit = cache.evaluate(
                        backend, inp["world_text"], tuple(candidates[c] for c in ids)
                    )
                    distributions.update(values)
                    requests.append(
                        {
                            "candidate_ids": ids,
                            "cache_key": key,
                            "cache_hit": hit,
                            "seconds": time.perf_counter() - tick,
                        }
                    )
                    calls += int(not hit)
                if set(distributions) != set(candidates):
                    raise ValueError("Baseline does not cover every candidate")
                result = (
                    Engine(
                        inp["program"],
                        candidates=inp["candidates"],
                        world=inp["world_text"],
                        backend=RecordedBackend(distributions),
                        config=config,
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
                    f"{index}/{len(inputs)} {inp['family']}: {len(requests)} model batches, {time.perf_counter() - program_start:.3f}s",
                    flush=True,
                )
        score = subprocess.run(
            [
                sys.executable,
                str(ROOT / "benchmarks/nl_logic_v1/score.py"),
                "--split",
                "dev",
                "--predictions",
                str((output / "predictions.jsonl").resolve()),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        metrics = json.loads(score.stdout)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        meta.update(
            status="complete",
            elapsed_seconds=time.perf_counter() - inference_start,
            model_backend_calls=calls,
        )
        (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        resources["status"] = "complete"
        print(json.dumps(metrics["overall"], indent=2))
    finally:
        resources["total_wall_seconds_including_init"] = time.perf_counter() - start
        resources["process_peak_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        (output / "resources.json").write_text(json.dumps(resources, indent=2) + "\n")


if __name__ == "__main__":
    main()
