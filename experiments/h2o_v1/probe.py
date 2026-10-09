"""Check native prompt/readout parity and deterministic CPU inference."""

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from relational_decisions import Candidate
from relational_decisions.h2o import H2OLightningBackend
from relational_decisions.prompts import WORLD_PREFIX, decode_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    backend = H2OLightningBackend()
    loaded = time.perf_counter()
    print(f"Loaded in {loaded - start:.2f}s", flush=True)
    world = "A report says Oak is ready. A separate report says Elm is not ready."
    candidates = (Candidate("oak", ("p",), ("n",), "Oak is ready", "Oak is not ready"),)
    tasks, requests = backend.requests(world, candidates)
    scores, checks = {}, []
    for task, (text, tokens, ids) in zip(tasks, requests):
        q = {
            "type": "choice",
            "instructions": task.instruction,
            "criteria": task.labels,
        }
        user, _, _, _ = backend.contract.question_parts(WORLD_PREFIX + world, q)
        rendered = (
            backend.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": backend.contract.system},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + backend.contract.prefill
        )
        assert text == rendered
        tick = time.perf_counter()
        hidden, selected = backend.hidden_and_logits(tokens, ids)
        elapsed = time.perf_counter() - tick
        _, repeat = backend.hidden_and_logits(tokens, ids)
        assert torch.equal(selected, repeat)
        # Independent full-vocabulary FP32 projection, chunked to bound RAM.
        with torch.inference_mode():
            head = backend.model.lm_head.weight
            full = torch.cat(
                [
                    torch.nn.functional.linear(hidden, head[i : i + 16384].float())[0]
                    for i in range(0, len(head), 16384)
                ]
            )
            logprobs = full.log_softmax(-1)[ids].tolist()
        native = backend.shim.format_answer(
            "choice",
            list(task.labels),
            list(task.labels.values()),
            logprobs,
            backend.contract.temperature,
            backend.contract.noul_floor,
        )["probabilities"]
        probabilities = dict(
            zip(
                task.labels,
                backend.shim.probabilities(
                    selected.tolist(),
                    backend.contract.temperature,
                ),
            )
        )
        delta = max(abs(native[k] - probabilities[k]) for k in native)
        assert delta < 1e-5, delta
        scores[task.name] = probabilities
        checks.append(
            {
                "task": task.name,
                "input_tokens": len(tokens),
                "label_token_ids": ids,
                "first_forward_seconds": elapsed,
                "repeat_bitwise_identical": True,
                "native_chat_template_identical": True,
                "full_vocab_native_readout_max_probability_difference": delta,
                "probabilities": probabilities,
            }
        )
        print(json.dumps(checks[-1]), flush=True)
    output = {
        "backend": backend.identity,
        "checks": checks,
        "assessments": decode_scores(tasks, scores),
        "scope": "CPU Transformers parity only; GPU vLLM numerical parity not tested",
        "wall_seconds_including_load": time.perf_counter() - start,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output["assessments"]), flush=True)


if __name__ == "__main__":
    main()
