"""Optional CPU OpenDecision cross-encoder adapter with lossless input checks."""

import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import distribution, version
from pathlib import Path

from .decisions import LABELS, Candidate
from .prompts import WORLD_PREFIX, build_tasks, decode_scores

DEFAULT_MODEL = "Tokz-labs/OpenDecision-Large"
DEFAULT_REVISION = "37ced3072592962b1e0cfdcc77eb511f8a3f7b54"
PROMPT = "binary-reports-v2"
STATE_TEMPLATE = "Question: {instruction}\n\n{world_prefix}{world}"
ANSWER_NAME = "answer"


def prepare_requests(world, candidates):
    tasks = build_tasks(candidates, PROMPT)
    texts = [
        STATE_TEMPLATE.format(
            instruction=t.instruction, world_prefix=WORLD_PREFIX, world=world
        )
        for t in tasks
    ]
    return tasks, texts


def check_input_lengths(encoder, config, max_positions, texts, labels):
    """Reject all four upstream truncation paths before calling the model."""
    tokenize = lambda text: encoder.tokenizer.encode(text, add_special_tokens=False)
    header = tokenize(ANSWER_NAME)
    if len(header) > config.max_header_tokens:
        raise ValueError("OpenDecision would truncate the decision header")
    option_lengths = []
    for label, description in labels.items():
        # Native serialization: [DEC] header [CAND] name :: description.
        length = 2 + len(header) + len(tokenize(f"{label} :: {description}"))
        if length > min(config.max_candidate_tokens, config.cross_candidate_tokens):
            raise ValueError("OpenDecision would truncate an answer option")
        option_lengths.append(length)
    state_lengths = [len(tokenize(text)) + 2 for text in texts]
    if any(length > config.max_state_tokens for length in state_lengths):
        raise ValueError("OpenDecision would truncate a question or world")
    # CrossEncoder.forward keeps at most max_positions - option_length - 3
    # tokens of an already [CLS]/[SEP]-wrapped state, dropping from the left.
    if any(s + c + 3 > max_positions for s in state_lengths for c in option_lengths):
        raise ValueError("OpenDecision would drop the start of a question/world")
    return max(state_lengths, default=0), max(option_lengths, default=0)


def decode_responses(tasks, policies):
    if len(tasks) != len(policies):
        raise ValueError("OpenDecision did not return exactly one row per question")
    return decode_scores(tasks, {t.name: p for t, p in zip(tasks, policies)})


class OpenDecisionBackend:
    def __init__(self, model=DEFAULT_MODEL, revision=None, *, threads=4, prompt=PROMPT):
        if prompt != PROMPT:
            raise ValueError(f"OpenDecision currently supports only {PROMPT}")
        if threads < 1:
            raise ValueError("Thread count must be positive")
        import opendecision
        import torch
        from huggingface_hub import HfApi, snapshot_download
        from opendecision import Candidate as Option
        from opendecision import Choice, OpenDecision

        revision = revision or (
            DEFAULT_REVISION
            if model == DEFAULT_MODEL
            else HfApi().model_info(model).sha
        )
        if len(revision) != 40 or any(
            c not in "0123456789abcdef" for c in revision.lower()
        ):
            revision = HfApi().model_info(model, revision=revision).sha
        path = snapshot_download(model, revision=revision)
        config = json.loads((Path(path) / "config.json").read_text())
        if config.get("architecture") != "cross_encoder":
            raise ValueError(
                "This adapter has been checked only for the cross_encoder architecture"
            )
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self.model = OpenDecision.from_pretrained(path, device="cpu")
        self.model.model.float().eval()
        self.model.precision = "fp32"
        template = Candidate(
            "candidate", ("p",), ("not_p",), "{positive}", "{negative}"
        )
        tasks = build_tasks([template], PROMPT)
        self.labels = tasks[0].labels
        self.spec = Choice([Option(k, k, v) for k, v in self.labels.items()])
        package_source = None
        direct = distribution("opendecision").read_text("direct_url.json")
        if direct:
            package_source = json.loads(direct).get("vcs_info", {}).get("commit_id")
        checkout = Path(opendecision.__file__).resolve().parents[2]
        if package_source is None and (checkout / ".git").exists():
            package_source = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
            ).strip()
        self._identity = {
            "backend": "opendecision",
            "adapter_version": "question-in-state-v1",
            "model": model,
            "revision": revision,
            "prompt_version": PROMPT,
            "world_prefix": WORLD_PREFIX,
            "task_templates": [asdict(t) for t in tasks],
            "state_template": STATE_TEMPLATE,
            "decision_name": ANSWER_NAME,
            "label_order": LABELS,
            "score_composition": "independent-positive-negative-product",
            "score_head": "policy",
            "activation": "softmax",
            "calibration": "none",
            "architecture": "cross_encoder",
            "cross_question_attention": False,
            "config": config,
            "dtype": "float32",
            "device": "cpu",
            "threads": threads,
            "question_batch_size": 8,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "opendecision_source_commit": package_source,
            "packages": {
                p: version(p)
                for p in ("opendecision", "torch", "transformers", "tokenizers")
            },
            "deterministic_algorithms": True,
            "truncation": "reject",
        }

    @property
    def identity(self):
        return self._identity

    def assess(self, world, candidates):
        if not candidates:
            return {}
        tasks, texts = prepare_requests(world, candidates)
        check_input_lengths(
            self.model.encoder,
            self.model.config,
            self.model.model.max_pos,
            texts,
            self.labels,
        )
        responses = self.model.decide(texts, {ANSWER_NAME: self.spec}, batch_size=8)
        return decode_responses(tasks, [r[ANSWER_NAME].policy for r in responses])
