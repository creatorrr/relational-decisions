"""Run the train-selected prompt on dev and record Linux process resources."""

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
    parser.add_argument("--model", choices=["340m", "1b"], required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_bytes = Path(__file__).with_name("protocol.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    selection_bytes = args.selection.read_bytes()
    selection = json.loads(selection_bytes)
    if (
        selection["status"] != "complete"
        or selection["protocol_sha256"] != hashlib.sha256(protocol_bytes).hexdigest()
    ):
        parser.error("Selection must come from the completed frozen train screen")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output must be a new directory")
    cache = args.output.with_name(args.output.name + "-cache")
    if cache.exists() and any(cache.iterdir()):
        parser.error("Cache directory must be new for uncached timing")
    model = protocol["selection_model" if args.model == "340m" else "comparison_model"]
    sys.argv = [
        "rd-bench",
        "--backend",
        "gliner",
        "--split",
        "dev",
        "--model",
        model["model"],
        "--revision",
        model["revision"],
        "--prompt",
        selection["selected_prompt"],
        "--schedule",
        "frontier",
        "--mode",
        "hard",
        "--batch-size",
        "4",
        "--search-quantum",
        "64",
        "--cache",
        str(cache),
        "--output",
        str(args.output),
    ]
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
