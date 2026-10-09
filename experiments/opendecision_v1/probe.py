"""Small deterministic/attention check; no benchmark gold or heldout access."""

import argparse
import json
import resource
import time
from pathlib import Path

from relational_decisions import Candidate
from relational_decisions.opendecision import OpenDecisionBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    backend = OpenDecisionBackend()
    model = backend.model.model
    masks = []

    def inspect_mask(module, positional, keywords):
        mask = keywords.get("attention_mask")
        if mask is None:
            mask = positional[1]
        # Check a single real row, excluding padding. The first token can
        # attend to every later real token in a bidirectional encoder.
        row = mask[0, 0].bool()
        real = row.diagonal().nonzero().flatten()
        masks.append(
            {
                "shape": list(mask.shape),
                "first_token_attends_to_last_real_token": bool(row[real[0], real[-1]]),
                "last_real_token_attends_to_first_token": bool(row[real[-1], real[0]]),
            }
        )

    hook = model.backbone.encoder.layer[0].attention.self.register_forward_pre_hook(
        inspect_mask, with_kwargs=True
    )
    world = "A report says Oak is ready. A separate report says Elm is not ready."
    candidates = tuple(
        Candidate(x, ("p", x), ("n", x), f"{x} is ready", f"{x} is not ready")
        for x in ("Oak", "Elm")
    )
    a = backend.assess(world, candidates)
    b = backend.assess(world, candidates)
    single = backend.assess(world, candidates[:1])
    hook.remove()
    delta = max(abs(a["Oak"][k] - single["Oak"][k]) for k in a["Oak"])
    assert a == b
    assert all(
        m["first_token_attends_to_last_real_token"]
        and m["last_real_token_attends_to_first_token"]
        for m in masks
    )
    assert delta < 1e-5
    result = {
        "backend": backend.identity,
        "parameters": sum(p.numel() for p in model.parameters()),
        "assessments": a,
        "repeat_bitwise_identical": a == b,
        "first_candidate_max_difference_with_second_candidate_removed": delta,
        "first_layer_attention_masks": masks,
        "interpretation": "Bidirectional attention within each option row; separate questions/options occupy independent batch rows. Policy softmax couples only options belonging to one question.",
        "wall_seconds_including_load": time.perf_counter() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
