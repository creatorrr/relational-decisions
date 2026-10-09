"""Check baseline reproducibility and independent-row batching before ablation."""

import argparse
import json
from pathlib import Path

from isolated import IsolatedGLiNERBackend, question_rows

from relational_decisions import Candidate
from relational_decisions.decisions import best_label, normalize
from relational_decisions.gliner import GLiNERBackend
from relational_decisions.prompts import decode_scores

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    protocol = json.loads(Path(__file__).with_name("protocol.json").read_text())
    backend = IsolatedGLiNERBackend(
        protocol["model"], protocol["revision"], layout="isolated-native"
    )
    inputs = [
        json.loads(line)
        for line in (ROOT / "benchmarks/nl_logic_v1/data/dev.inputs.jsonl")
        .read_text()
        .splitlines()
    ]
    archived = json.loads(
        (ROOT / (protocol["baseline"] + ".traces.jsonl")).read_text().splitlines()[0]
    )
    inp = next(row for row in inputs if row["id"] == archived["id"])
    by_id = {c["id"]: Candidate(**c) for c in inp["candidates"]}
    candidates = tuple(by_id[c] for c in archived["trace"][0]["candidate_ids"])
    observed = {
        cid: normalize(scores)
        for cid, scores in GLiNERBackend.assess(
            backend, inp["world_text"], candidates
        ).items()
    }
    expected = {cid: normalize(archived["distributions"][cid]) for cid in observed}
    baseline_delta = max(
        abs(float(observed[c][k] - expected[c][k]))
        for c in observed
        for k in observed[c]
    )
    assert baseline_delta == 0, baseline_delta

    from gliner2.classification import ClassificationSchema
    from gliner2.classification.compiler import compile_schema

    world = "A report says Oak is ready. A separate report says Elm is not ready."
    candidates = tuple(
        Candidate(x, ("p", x), ("n", x), f"{x} is ready", f"{x} is not ready")
        for x in ("Oak", "Elm")
    )
    batched = backend.assess(world, candidates)
    repeat = backend.assess(world, candidates)
    rows = question_rows(world, candidates, "isolated-native")
    scores = {}
    for task, name, instruction, text in rows:
        schema = ClassificationSchema()
        schema.single(name, task.labels, instruction=instruction, activation="softmax")
        one = backend.classifier.score(text, compile_schema(schema))
        scores[task.name] = {
            label: one.probability(name, label) for label in task.labels
        }
    serial = decode_scores([r[0] for r in rows], scores)
    delta = max(abs(batched[c][k] - serial[c][k]) for c in batched for k in batched[c])
    assert batched == repeat
    assert delta < 1e-5, delta
    assert all(best_label(batched[c]) == best_label(serial[c]) for c in batched)
    result = {
        "backend": backend.identity,
        "baseline_program_id": inp["id"],
        "baseline_first_group_reproduced_exactly": baseline_delta == 0,
        "baseline_max_probability_difference": baseline_delta,
        "isolated_repeat_bitwise_identical": batched == repeat,
        "isolated_batched_vs_serial_max_category_weight_difference": delta,
        "isolated_batched_vs_serial_hard_labels_identical": True,
        "isolated_smoke_assessments": batched,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
