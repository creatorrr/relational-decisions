"""Verify the replay/evaluation plumbing using archived scores, without ML."""

import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import run_dev


def main():
    root = Path(__file__).resolve().parents[2]
    prefix = root / "experiments/prompt_v2/decide-340m-dev"
    inputs = run_dev.read_rows(root / "benchmarks/nl_logic_v1/data/dev.inputs.jsonl")
    traces = run_dev.read_rows(Path(str(prefix) + ".traces.jsonl"))
    expected = run_dev.read_rows(Path(str(prefix) + ".predictions.jsonl"))
    distributions = {
        row["world_text"]: traces[pid]["distributions"] for pid, row in inputs.items()
    }
    assert len(distributions) == len(inputs)

    class ArchivedScores:
        def __init__(self, *args, **kwargs):
            self.identity = {"backend": "replay-validation-only"}

        def assess(self, world, candidates):
            return {c.id: distributions[world][c.id] for c in candidates}

    with tempfile.TemporaryDirectory() as temp:
        output = Path(temp) / "run"
        command = ["run_dev.py", "--layout", "isolated-native", "--output", str(output)]
        with (
            patch.object(run_dev, "IsolatedGLiNERBackend", ArchivedScores),
            patch.object(sys, "argv", command),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            run_dev.main()
        observed = run_dev.read_rows(output / "predictions.jsonl")
        assert observed == expected
        replay = run_dev.read_rows(output / "traces.jsonl")
        for pid, row in replay.items():
            assert [r["candidate_ids"] for r in row["replayed_model_requests"]] == [
                r["candidate_ids"] for r in traces[pid]["trace"]
            ]
    print(
        json.dumps(
            {
                "programs_reproduced_exactly": len(expected),
                "assessments": sum(len(r["assessments"]) for r in expected.values()),
                "queries": sum(
                    len(r["query_probabilities"]) for r in expected.values()
                ),
                "model_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
