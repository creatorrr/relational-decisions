"""Run only open benchmark splits; write predictions, traces, and scored metrics."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from .decisions import DecisionCache, OracleBackend
from .engine import Engine, EngineConfig
from .prompts import DEFAULT_PROMPT, PROMPTS


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", type=Path, default=Path("benchmarks/nl_logic_v1")
    )
    parser.add_argument("--split", choices=["train", "dev"], default="dev")
    parser.add_argument(
        "--backend",
        choices=["oracle", "gliner", "opendecision", "openjev", "h2o"],
        default="oracle",
    )
    parser.add_argument(
        "--schedule", choices=["frontier", "full", "single"], default="frontier"
    )
    parser.add_argument("--mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--search-quantum", type=int, default=64)
    parser.add_argument("--model", help="Backend's pinned default model if omitted")
    parser.add_argument("--revision")
    parser.add_argument(
        "--prompt", choices=PROMPTS, help="Backend's default prompt if omitted"
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("runs/decision-cache"))
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; choose a new run directory")
    args.output.mkdir(parents=True, exist_ok=True)
    inputs_path = args.benchmark / "data" / f"{args.split}.inputs.jsonl"
    inputs = load_rows(inputs_path)
    config = EngineConfig(
        schedule=args.schedule,
        mode=args.mode,
        batch_size=args.batch_size,
        search_quantum=args.search_quantum,
    )
    oracle = None
    if args.backend == "oracle":
        oracle = {
            r["id"]: r["assessments"]
            for r in load_rows(args.benchmark / "data" / f"{args.split}.gold.jsonl")
        }
        backend = None
    elif args.backend == "gliner":
        from .gliner import GLiNERBackend

        backend = GLiNERBackend(
            args.model or "fastino/gliner2.5-small-v1",
            args.revision,
            threads=args.threads,
            max_tokens=args.max_tokens,
            prompt=args.prompt or DEFAULT_PROMPT,
        )
    elif args.backend == "opendecision":
        from .opendecision import DEFAULT_MODEL, PROMPT, OpenDecisionBackend

        backend = OpenDecisionBackend(
            args.model or DEFAULT_MODEL,
            args.revision,
            threads=args.threads,
            prompt=args.prompt or PROMPT,
        )
    elif args.backend == "openjev":
        from .openjev import DEFAULT_MODEL, PROMPT, OpenJevBackend

        backend = OpenJevBackend(
            args.model or DEFAULT_MODEL,
            args.revision,
            threads=args.threads,
            prompt=args.prompt or PROMPT,
        )
    else:
        from .h2o import DEFAULT_MODEL, DEFAULT_REVISION, PROMPT, H2OLightningBackend

        if args.model not in (None, DEFAULT_MODEL) or args.revision not in (
            None,
            DEFAULT_REVISION,
        ):
            parser.error("H2O supports only the reviewed, pinned model snapshot")
        if args.prompt not in (None, PROMPT):
            parser.error(f"H2O supports only {PROMPT}")
        backend = H2OLightningBackend(threads=args.threads)
    metadata = {
        "created": datetime.now(UTC).isoformat(),
        "split": args.split,
        "backend": backend.identity if backend else {"backend": "oracle-v1"},
        "config": asdict(config),
        "complete_assessments": True,
        "input_sha256": hashlib.sha256(inputs_path.read_bytes()).hexdigest(),
        "programs": len(inputs),
        "status": "running",
    }
    metadata_path = args.output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    cache = DecisionCache(args.cache)
    start = time.perf_counter()
    summaries = []
    predictions_path = args.output / "predictions.jsonl"
    with (
        predictions_path.open("w") as predictions,
        (args.output / "traces.jsonl").open("w") as traces,
    ):
        for index, inp in enumerate(inputs, 1):
            actual_backend = (
                OracleBackend(oracle[inp["id"]]) if oracle is not None else backend
            )
            result = Engine(
                inp["program"],
                candidates=inp["candidates"],
                world=inp["world_text"],
                backend=actual_backend,
                config=config,
                cache=cache,
            ).run(complete_assessments=True)
            encoded = result.to_dict()
            predictions.write(
                json.dumps(
                    {
                        "id": inp["id"],
                        "assessments": encoded["assessments"],
                        "query_probabilities": encoded["query_probabilities"],
                    }
                )
                + "\n"
            )
            traces.write(json.dumps({"id": inp["id"], **encoded}) + "\n")
            predictions.flush()
            traces.flush()
            summaries.append(result.stats)
            print(
                f"{index}/{len(inputs)} {inp['family']}: {result.stats['batches']} batches, "
                f"{result.stats['seconds']:.3f}s",
                flush=True,
            )
    # Model execution receives input records only. Gold is read by the separate
    # scorer after predictions are complete, never by the neural adapter.
    scored = subprocess.run(
        [
            sys.executable,
            str(args.benchmark / "score.py"),
            "--split",
            args.split,
            "--predictions",
            str(predictions_path.resolve()),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    metrics = json.loads(scored.stdout)
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    metadata.update(
        {
            "status": "complete",
            "elapsed_seconds": time.perf_counter() - start,
            "backend_calls": sum(s["backend_calls"] for s in summaries),
            "batches": sum(s["batches"] for s in summaries),
            "max_diagram_nodes": max(s["diagram_nodes"] for s in summaries),
            "max_tables": max(s["tables"] for s in summaries),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "overall": metrics["overall"],
                "run": str(args.output),
                "metadata": metadata,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
