"""Compare the same eight diagnostic programs; never opens heldout."""

import json
from pathlib import Path

from run_dev import read_rows
from run_quick import score_subset

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    inputs = dict(
        list(read_rows(ROOT / "benchmarks/nl_logic_v1/data/dev.inputs.jsonl").items())[
            :8
        ]
    )
    gold = read_rows(ROOT / "benchmarks/nl_logic_v1/data/dev.gold.jsonl")
    prefixes = {
        "gliner-340m-shared": HERE.parent / "prompt_v2/decide-340m-dev",
        "gliner-340m-isolated-native": HERE / "gliner-isolated-native-partial",
        "gliner-340m-isolated-state": HERE / "gliner-state-quick8",
        "gliner-1b-shared": HERE.parent / "prompt_v2/decide-1b-dev",
        "opendecision-isolated-state": HERE.parent / "opendecision_v1/opendecision-dev",
        "openjev-isolated-native": HERE.parent / "openjev_v1/openjev-quick8",
    }
    results = {
        name: score_subset(
            inputs, read_rows(Path(str(prefix) + ".predictions.jsonl")), gold
        )
        for name, prefix in prefixes.items()
    }
    report = {
        "scope": "Same first 8 dev programs; quick diagnostic, no generalization claim",
        "input_ids": list(inputs),
        "results": results,
        "heldout_opened": False,
    }
    (HERE / "quick-comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v["overall"] for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
