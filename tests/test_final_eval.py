"""Exercise the final-evaluation runner using public dev fixtures only."""

import contextlib
import importlib.util
import io
import json
import sys
import tarfile
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from relational_decisions.decisions import OracleBackend
from relational_decisions.engine import EngineConfig

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "final_eval_fixture", ROOT / "experiments/h2o_v1/final_eval.py"
)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class FinalEvaluationTests(unittest.TestCase):
    def test_predictions_are_sealed_before_gold_and_resume_keeps_completed_work(self):
        self.exercise_runner(gguf=False)

    def test_gguf_server_cleanup_and_resources_survive_resume(self):
        self.exercise_runner(gguf=True)

    def exercise_runner(self, *, gguf):
        data = ROOT / "benchmarks/nl_logic_v1/data"
        all_inputs = DRIVER.parse_rows((data / "dev.inputs.jsonl").read_bytes())
        group = next(iter(all_inputs.values()))["group_id"]
        inputs = {
            pid: row for pid, row in all_inputs.items() if row["group_id"] == group
        }
        gold = {
            pid: row
            for pid, row in DRIVER.parse_rows(
                (data / "dev.gold.jsonl").read_bytes()
            ).items()
            if pid in inputs
        }
        labels = {
            row["world_text"]: gold[pid]["assessments"] for pid, row in inputs.items()
        }

        class PublicFixtureBackend:
            identity: ClassVar[dict] = {
                "backend": "public-dev-fixture",
                "label_order": ("supported", "refuted", "both", "unknown"),
            }
            closed: ClassVar[int] = 0

            def __init__(self, *args, **kwargs):
                pass

            def assess(self, world, candidates):
                return OracleBackend(labels[world]).assess(world, candidates)

            def resources(self):
                return {"server_peak_rss_kib": 1234}

            def close(self):
                type(self).closed += 1

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            archive, output = folder / "fixture.tar.gz", folder / "output"
            with tarfile.open(archive, "w:gz") as bundle:
                for kind, rows in (("inputs", inputs), ("gold", gold)):
                    raw = "".join(json.dumps(r) + "\n" for r in rows.values()).encode()
                    info = tarfile.TarInfo(f"heldout.{kind}.jsonl")
                    info.size = len(raw)
                    bundle.addfile(info, io.BytesIO(raw))
            protocol = folder / "protocol.json"
            DRIVER.write_json(
                protocol,
                {
                    "archive": str(archive),
                    "archive_sha256": DRIVER.sha(archive),
                    "source_sha256": {},
                    "threads": 4,
                    **({"gguf_model": "h2o-q8"} if gguf else {}),
                    "backend_identity": PublicFixtureBackend.identity,
                    "engine_config": asdict(EngineConfig()),
                    "counts": {
                        "programs": len(inputs),
                        "assessments": sum(
                            len(r["candidates"]) for r in inputs.values()
                        ),
                        "queries": sum(
                            len(r["query_probabilities"]) for r in gold.values()
                        ),
                    },
                },
            )
            original_read = DRIVER.read_member
            reads = []

            def checked_read(path, name):
                if name.endswith("gold.jsonl"):
                    seal = json.loads(
                        (output / "predictions-complete.json").read_text()
                    )
                    self.assertEqual(
                        seal["predictions_sha256"],
                        DRIVER.sha(output / "predictions.jsonl"),
                    )
                    self.assertEqual(seal["programs"], len(inputs))
                reads.append(name)
                return original_read(path, name)

            original_run = DRIVER.run_program
            calls = 0

            def interrupted_run(*args):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("Simulated technical interruption")
                return original_run(*args)

            command = [
                "final_eval.py",
                "--protocol",
                str(protocol),
                "--output",
                str(output),
            ]
            with (
                patch.object(
                    DRIVER, "H2OLightningBackend", lambda **kw: PublicFixtureBackend()
                ),
                patch.object(DRIVER, "GGUFBackend", PublicFixtureBackend)
                if gguf
                else contextlib.nullcontext(),
                patch.object(DRIVER, "read_member", checked_read),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                with (
                    patch.object(sys, "argv", command),
                    patch.object(DRIVER, "run_program", interrupted_run),
                    self.assertRaisesRegex(RuntimeError, "Simulated"),
                ):
                    DRIVER.main()
                self.assertEqual(reads, ["heldout.inputs.jsonl"])
                before = (output / "checkpoints/001.json").read_bytes()
                with (
                    patch.object(sys, "argv", command + ["--resume"]),
                    patch.object(DRIVER, "run_program", wraps=original_run) as resumed,
                ):
                    DRIVER.main()
                self.assertEqual(resumed.call_count, 1)
                self.assertEqual(before, (output / "checkpoints/001.json").read_bytes())
                metrics = json.loads((output / "metrics.json").read_text())
                self.assertEqual(metrics["overall"]["assessment_accuracy"], 1.0)
                self.assertEqual(metrics["overall"]["query_probability_mae"], 0.0)
                self.assertEqual(reads.count("heldout.gold.jsonl"), 1)
                if gguf:
                    self.assertEqual(PublicFixtureBackend.closed, 2)
                    resources = json.loads((output / "resources.json").read_text())
                    self.assertEqual(resources["server_peak_rss_kib"], 1234)


if __name__ == "__main__":
    unittest.main()
