"""Verify batched inference against OpenJev's public API and check attention."""

import argparse
import json
import resource
import time
from pathlib import Path

from relational_decisions import Candidate
from relational_decisions.decisions import best_label
from relational_decisions.openjev import OpenJevBackend, option_texts
from relational_decisions.prompts import decode_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    backend = OpenJevBackend()
    masks = []

    def inspect_mask(module, positional, keywords):
        mask = keywords.get("attention_mask")
        if mask is None:
            mask = positional[1]
        row = mask[0, 0].bool()
        real = row.diagonal().nonzero().flatten()
        masks.append(
            {
                "shape": list(mask.shape),
                "first_to_last": bool(row[real[0], real[-1]]),
                "last_to_first": bool(row[real[-1], real[0]]),
            }
        )

    hook = backend.runtime.model.backbone.encoder.layer[
        0
    ].attention.self.register_forward_pre_hook(inspect_mask, with_kwargs=True)
    world = "A report says Oak is ready. A separate report says Elm is not ready."
    candidates = tuple(
        Candidate(x, ("p", x), ("n", x), f"{x} is ready", f"{x} is not ready")
        for x in ("Oak", "Elm")
    )
    batched = backend.assess(world, candidates)
    repeat = backend.assess(world, candidates)
    tasks, requests = backend.requests(world, candidates)
    scores = {}
    for task, (state, _) in zip(tasks, requests):
        options = option_texts(task)
        result = backend.runtime.decide(
            state,
            [{"type": "choice", "instructions": task.instruction, "options": options}],
        )[0]
        scores[task.name] = {
            label: result["probabilities"][text]
            for label, text in zip(task.labels, options)
        }
    official = decode_scores(tasks, scores)
    hook.remove()
    delta = max(
        abs(batched[c][k] - official[c][k]) for c in batched for k in batched[c]
    )
    assert batched == repeat
    assert delta < 1e-5, delta
    assert all(best_label(batched[c]) == best_label(official[c]) for c in batched)
    assert all(m["first_to_last"] and m["last_to_first"] for m in masks)
    output = {
        "backend": backend.identity,
        "parameters": sum(p.numel() for p in backend.runtime.model.parameters()),
        "repeat_bitwise_identical": batched == repeat,
        "native_api_vs_batched_max_category_weight_difference": delta,
        "native_api_vs_batched_hard_labels_identical": True,
        "first_layer_attention_masks": masks,
        "assessments": batched,
        "wall_seconds_including_load": time.perf_counter() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
