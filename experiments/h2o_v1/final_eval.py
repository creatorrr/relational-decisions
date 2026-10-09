"""One frozen heldout evaluation; gold is read only after predictions are sealed."""

import argparse
import fcntl
import hashlib
import json
import resource
import subprocess
import sys
import tarfile
import time
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from relational_decisions.decisions import DecisionCache
from relational_decisions.engine import Engine, EngineConfig
from relational_decisions.h2o import H2OLightningBackend

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmarks/nl_logic_v1"))
from logic import solve
from score import evaluate


def now():
    return datetime.now(UTC).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parse_rows(raw):
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    result = {row["id"]: row for row in rows}
    if len(rows) != len(result):
        raise ValueError("Duplicate program IDs")
    return result


def read_member(archive, name):
    # Read the named regular file directly, without extracting archive paths.
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember(name)
        if not member.isfile():
            raise ValueError("Expected a regular JSONL archive member")
        return bundle.extractfile(member).read()


def check_frozen_files(protocol):
    for name, expected in protocol["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    archive = ROOT / protocol["archive"]
    if sha(archive) != protocol["archive_sha256"]:
        raise ValueError("Heldout archive checksum differs from the frozen protocol")
    return archive


def run_program(inp, backend, config, cache):
    return (
        Engine(
            inp["program"],
            candidates=inp["candidates"],
            world=inp["world_text"],
            backend=backend,
            config=config,
            cache=cache,
        )
        .run(complete_assessments=True)
        .to_dict()
    )


def verify_and_score(inputs, gold, predictions):
    metrics = evaluate(inputs, gold, predictions)
    verified = 0
    for pid, inp in inputs.items():
        # Independently validate both the frozen gold and the submitted engine output.
        truth = solve(inp["program"], gold[pid]["oracle_facts"])
        assert truth["query_probabilities"] == gold[pid]["query_probabilities"]
        assert truth["evidence_probability"] == gold[pid]["evidence_probability"]
        facts = []
        for candidate in inp["candidates"]:
            label = predictions[pid]["assessments"][candidate["id"]]
            if label in ("supported", "both"):
                facts.append(candidate["atom"])
            if label in ("refuted", "both"):
                facts.append(candidate["negative_atom"])
        reference = solve(inp["program"], facts)["query_probabilities"]
        assert {q: Fraction(p) for q, p in reference.items()} == {
            q: Fraction(p) for q, p in predictions[pid]["query_probabilities"].items()
        }
        verified += len(reference)
    metrics["verified_reference_queries"] = verified
    metrics["verified_ground_truth_programs"] = len(inputs)
    metrics["scope"] = "Final heldout evaluation of the frozen selected H2O system"
    return metrics


class ProgressBackend:
    """Record work completed without logging scores or changing model requests."""

    def __init__(self, backend, path):
        self.backend, self.path = backend, path
        self.program_index = 0
        self.candidates_this_program = 0

    @property
    def identity(self):
        return self.backend.identity

    def assess(self, world, candidates):
        result = self.backend.assess(world, candidates)
        self.candidates_this_program += len(candidates)
        write_json(
            self.path,
            {
                "updated_utc": now(),
                "program_index": self.program_index,
                "newly_evaluated_candidates_this_program": self.candidates_this_program,
            },
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", type=Path, default=Path(__file__).with_name("final-protocol.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Technical recovery with the identical frozen configuration",
    )
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        parser.error(
            "Use a fresh output directory; --resume is only for technical recovery"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    # Prevent two workers from running the same evaluation directory.
    with (args.output / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_locked(args)


def run_locked(args):
    protocol = json.loads(args.protocol.read_text())
    archive = check_frozen_files(protocol)
    metadata_path = args.output / "metadata.json"
    if args.resume:
        metadata = json.loads(metadata_path.read_text())
        if metadata["protocol_sha256"] != sha(args.protocol):
            raise ValueError("Cannot resume with a different protocol")
        if metadata["status"] == "complete":
            raise ValueError("The heldout evaluation already completed")
        for previous in metadata["attempts"]:
            if "wall_seconds" not in previous:
                previous.update(
                    status="interrupted-without-resource-summary",
                    wall_seconds=None,
                    peak_rss_kib=None,
                )
    else:
        metadata = {
            "status": "initializing",
            "split": "heldout",
            "started_utc": now(),
            "protocol_sha256": sha(args.protocol),
            "archive_sha256": sha(archive),
            "implementation_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "gold_read": False,
            "attempts": [],
        }
    attempt = {"started_utc": now(), "status": "failed", "resume": args.resume}
    metadata["attempts"].append(attempt)
    write_json(metadata_path, metadata)
    start = time.perf_counter()
    try:
        backend = H2OLightningBackend(threads=protocol["threads"])
        # JSON normalizes tuple-valued fields such as label_order to lists.
        identity = json.loads(json.dumps(backend.identity))
        if identity != protocol["backend_identity"]:
            raise ValueError(
                "Runtime backend identity differs from the frozen selection"
            )
        metadata["backend"] = identity
        metadata["config"] = protocol["engine_config"]
        # This is the first access to heldout inputs. No gold member is read here.
        input_bytes = read_member(archive, "heldout.inputs.jsonl")
        inputs = parse_rows(input_bytes)
        if len(inputs) != protocol["counts"]["programs"]:
            raise ValueError("Heldout program count differs from the manifest")
        if (
            sum(len(inp["candidates"]) for inp in inputs.values())
            != protocol["counts"]["assessments"]
        ):
            raise ValueError("Heldout assessment count differs from the manifest")
        if (
            metadata.get("input_sha256", hashlib.sha256(input_bytes).hexdigest())
            != hashlib.sha256(input_bytes).hexdigest()
        ):
            raise ValueError("Heldout inputs changed during resume")
        metadata.update(
            status="predicting",
            input_sha256=hashlib.sha256(input_bytes).hexdigest(),
            programs=len(inputs),
            inputs_opened_utc=metadata.get("inputs_opened_utc", now()),
        )
        write_json(metadata_path, metadata)
        config = EngineConfig(**protocol["engine_config"])
        tracking = ProgressBackend(backend, args.output / "progress.json")
        cache = DecisionCache(args.output / "decision-cache")
        checkpoints = args.output / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        rows = []
        for index, (pid, inp) in enumerate(inputs.items(), 1):
            checkpoint = checkpoints / f"{index:03}.json"
            if checkpoint.exists():
                row = json.loads(checkpoint.read_text())
                if row["id"] != pid:
                    raise ValueError("Checkpoint input order mismatch")
            else:
                tracking.program_index, tracking.candidates_this_program = index, 0
                row = {"id": pid, **run_program(inp, tracking, config, cache)}
                write_json(checkpoint, row)
            rows.append(row)
            metadata["programs_complete"] = index
            write_json(metadata_path, metadata)
            print(
                f"{index}/{len(inputs)} complete; {row['stats']['seconds']:.2f}s; elapsed {time.perf_counter() - start:.1f}s",
                flush=True,
            )
        for name, records in (
            (
                "predictions.jsonl",
                [
                    {k: r[k] for k in ("id", "assessments", "query_probabilities")}
                    for r in rows
                ],
            ),
            ("traces.jsonl", rows),
        ):
            temporary = args.output / (name + ".tmp")
            temporary.write_text("".join(json.dumps(r) + "\n" for r in records))
            temporary.replace(args.output / name)
        predictions_path = args.output / "predictions.jsonl"
        seal = {
            "completed_utc": now(),
            "programs": len(rows),
            "predictions_sha256": sha(predictions_path),
        }
        write_json(args.output / "predictions-complete.json", seal)
        metadata.update(status="scoring", **seal)
        write_json(metadata_path, metadata)
        # All predictions are durable and checksummed before opening gold.
        gold_bytes = read_member(archive, "heldout.gold.jsonl")
        metadata.update(
            gold_read=True,
            gold_opened_utc=now(),
            gold_sha256=hashlib.sha256(gold_bytes).hexdigest(),
        )
        write_json(metadata_path, metadata)
        metrics = verify_and_score(
            inputs, parse_rows(gold_bytes), parse_rows(predictions_path.read_bytes())
        )
        if metrics["overall"]["queries"] != protocol["counts"]["queries"]:
            raise ValueError("Heldout query count differs from the manifest")
        write_json(args.output / "metrics.json", metrics)
        metadata["status"] = attempt["status"] = "complete"
        metadata["finished_utc"] = now()
        print(json.dumps(metrics, indent=2), flush=True)
    except BaseException:
        metadata["status"] = "failed"
        raise
    finally:
        attempt.update(
            finished_utc=now(),
            wall_seconds=time.perf_counter() - start,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
        write_json(metadata_path, metadata)
        write_json(
            args.output / "resources.json",
            {
                "status": attempt["status"],
                "total_attempt_wall_seconds": sum(
                    a["wall_seconds"] or 0 for a in metadata["attempts"]
                ),
                "peak_rss_kib": max(
                    a["peak_rss_kib"] or 0 for a in metadata["attempts"]
                ),
                "attempts": len(metadata["attempts"]),
                "resource_totals_complete": all(
                    a["wall_seconds"] is not None for a in metadata["attempts"]
                ),
            },
        )


if __name__ == "__main__":
    main()
