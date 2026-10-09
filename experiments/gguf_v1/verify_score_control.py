import json, time
from pathlib import Path
from fractions import Fraction
from relational_decisions import Candidate
from relational_decisions.gguf import GGUFBackend, label_probabilities

root = Path(__file__).resolve().parents[2]
start = time.perf_counter()
backend = GGUFBackend("h2o-q8")
try:
    inp = json.loads(
        (root / "benchmarks/nl_logic_v1/data/dev.inputs.jsonl")
        .read_text()
        .splitlines()[0]
    )
    old = json.loads(
        (root / "experiments/gguf_v1/h2o-q8-quick8.traces.jsonl")
        .read_text()
        .splitlines()[0]
    )
    candidates = {c["id"]: Candidate(**c) for c in inp["candidates"]}
    results = {}
    for request in old["replayed_model_requests"]:
        results.update(
            backend.assess(
                inp["world_text"],
                tuple(candidates[i] for i in request["candidate_ids"]),
            )
        )
    differences = [
        abs(float(Fraction(old["distributions"][cid][label])) - prob)
        for cid, values in results.items()
        for label, prob in values.items()
    ]
    new_labels = {cid: max(values, key=values.get) for cid, values in results.items()}
    assert new_labels == old["assessments"]
    assert max(differences) < 1e-6
    _, requests = backend.renderer.requests(
        inp["world_text"], [next(iter(candidates.values()))]
    )
    _, tokens, ids = requests[0]
    byte_id = backend.renderer.tokenizer.convert_tokens_to_ids("ä")
    payload = {
        "prompt": tokens,
        "n_predict": 1,
        "temperature": -1,
        "n_probs": 128,
        "post_sampling_probs": False,
        "cache_prompt": False,
        "seed": 0,
        "samplers": [],
    }
    baseline = backend.request("completion", payload)
    incomplete = backend.request(
        "completion", {**payload, "logit_bias": [[byte_id, 1000]]}
    )
    constrained = backend.request(
        "completion",
        {
            **payload,
            "logit_bias": [[byte_id, 1000]],
            "grammar": 'root ::= " A"',
            "samplers": ["temperature"],
        },
    )
    assert not incomplete.get("completion_probabilities"), incomplete.keys()
    p1 = label_probabilities(baseline, ids, 0.8)
    p2 = label_probabilities(constrained, ids, 0.8)
    assert p1 == p2, (p1, p2)
    result = {
        "public_dev_program": inp["id"],
        "candidate_count": len(results),
        "all_hard_decisions_preserved": True,
        "maximum_distribution_difference": max(differences),
        "incomplete_utf8_token_id": byte_id,
        "incomplete_utf8_omits_probability_record": True,
        "ascii_grammar_restores_record": True,
        "raw_label_scores_bitwise_equal_despite_sampler_bias": True,
        "seconds": time.perf_counter() - start,
        "backend": backend.identity,
    }
    (root / "experiments/gguf_v1/score-control-check.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        json.dumps({k: v for k, v in result.items() if k != "backend"}, indent=2),
        flush=True,
    )
finally:
    backend.close()
