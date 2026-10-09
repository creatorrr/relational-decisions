"""Run the frozen OpenDecision comparison with a fresh cache and resource record."""

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

from relational_decisions.benchmark import main as benchmark_main

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = Path(__file__).with_name("protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    selection_bytes = (
        ROOT / "experiments/prompt_v2/train-screen/selection.json"
    ).read_bytes()
    if (
        hashlib.sha256(selection_bytes).hexdigest()
        != protocol["prior_selection_sha256"]
    ):
        parser.error("Frozen prior selection hash does not match")
    if json.loads(selection_bytes)["selected_prompt"] != protocol["prompt"]:
        parser.error("Prompt does not match the prior train selection")
    cache = args.output.with_name(args.output.name + "-cache")
    for path in (args.output, cache):
        if path.exists() and any(path.iterdir()):
            parser.error("Output and cache must be new directories")
    sys.argv = [
        "rd-bench",
        "--backend",
        "opendecision",
        "--cache",
        str(cache),
        "--output",
        str(args.output),
    ]
    for key in (
        "model",
        "revision",
        "split",
        "prompt",
        "schedule",
        "mode",
        "batch_size",
        "search_quantum",
        "threads",
    ):
        sys.argv.extend(["--" + key.replace("_", "-"), str(protocol[key])])
    measurements = {
        "command": sys.argv,
        "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "runtime_source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((ROOT / "src/relational_decisions").glob("*.py"))
        },
        "status": "failed",
    }
    start = time.perf_counter()
    try:
        benchmark_main()
        measurements["status"] = "complete"
    finally:
        measurements["total_wall_seconds_including_init"] = time.perf_counter() - start
        measurements["process_peak_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "resources.json").write_text(
            json.dumps(measurements, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
