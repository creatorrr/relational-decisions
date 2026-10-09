"""Pinned OpenJev implementation, with independent questions in native batch rows."""

import hashlib
import importlib
import importlib.util
import platform
import sys
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .decisions import LABELS, Candidate
from .prompts import WORLD_PREFIX, build_tasks, decode_scores

DEFAULT_MODEL = "com-kotobalabs/open-jev-deberta-v3-large"
DEFAULT_REVISION = "188ee67a5c93122b916e5acd5bdb0cb3623e380a"
PROMPT = "binary-reports-v2"


def option_texts(task):
    return [f"{label} :: {description}" for label, description in task.labels.items()]


def check_state_length(tokenizer, state, max_state_tokens):
    length = len(tokenizer.encode(state, add_special_tokens=False))
    if length > max_state_tokens:
        raise ValueError(
            f"OpenJev would truncate world state: {length} > {max_state_tokens}"
        )
    return length


def load_snapshot_module(path, revision):
    """Load reviewed snapshot code under a private namespace, without sys.path edits."""
    namespace = "_relational_openjev_" + revision
    package = Path(path) / "typed_decisions"
    if namespace not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            namespace,
            package / "__init__.py",
            submodule_search_locations=[str(package)],
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[namespace] = module
        spec.loader.exec_module(module)
    return importlib.import_module(namespace + ".open_jev")


class OpenJevBackend:
    def __init__(self, model=DEFAULT_MODEL, revision=None, *, threads=4, prompt=PROMPT):
        if model != DEFAULT_MODEL or (
            revision is not None and revision != DEFAULT_REVISION
        ):
            raise ValueError("Only the reviewed, pinned OpenJev snapshot is supported")
        if prompt != PROMPT:
            raise ValueError(f"OpenJev supports only {PROMPT}")
        if threads < 1:
            raise ValueError("Thread count must be positive")
        import torch
        from huggingface_hub import snapshot_download

        revision = DEFAULT_REVISION
        path = Path(snapshot_download(model, revision=revision))
        api = load_snapshot_module(path, revision)
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self.runtime = api.OpenJev.from_pretrained(str(path), device="cpu")
        self.runtime.model.float().eval()
        self.question_type = api.Question
        template = Candidate(
            "candidate", ("p",), ("not_p",), "{positive}", "{negative}"
        )
        self._identity = {
            "backend": "openjev",
            "adapter_version": "isolated-native-questions-v1",
            "model": model,
            "revision": revision,
            "prompt_version": PROMPT,
            "world_prefix": WORLD_PREFIX,
            "task_templates": [asdict(t) for t in build_tasks([template], PROMPT)],
            "label_order": LABELS,
            "score_composition": "independent-positive-negative-product",
            "option_template": "{label} :: {description}",
            "question_type": "choice",
            "questions_per_attention_sequence": 1,
            "question_batch_size": 8,
            "question_layout": "native schema instructions",
            "activation": "softmax",
            "temperature": self.runtime.model.temperature,
            "calibration": "published temperature; no calibration on this benchmark",
            "config": self.runtime.config,
            "published_source_sha256": {
                str(p.relative_to(path)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((path / "typed_decisions").glob("*.py"))
            },
            "dtype": "float32",
            "device": "cpu",
            "threads": threads,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "packages": {
                p: version(p)
                for p in (
                    "torch",
                    "transformers",
                    "tokenizers",
                    "safetensors",
                    "huggingface-hub",
                )
            },
            "deterministic_algorithms": True,
            "truncation": "reject",
        }

    @property
    def identity(self):
        return self._identity

    def requests(self, world, candidates):
        tasks = build_tasks(candidates, PROMPT)
        state = WORLD_PREFIX + world
        check_state_length(self.runtime.tok, state, self.runtime.collator.max_state)
        # The native dataclass requires a gold index for its training collator.
        # This constant placeholder is never passed into model.forward.
        questions = [
            self.question_type(t.name, "choice", t.instruction, option_texts(t), 0)
            for t in tasks
        ]
        return tasks, [(state, [q]) for q in questions]

    def assess(self, world, candidates):
        import torch

        tasks, requests = self.requests(world, candidates)
        policies = []
        with torch.inference_mode():
            for offset in range(0, len(requests), 8):
                # Native collator raises on joint inputs over its 512-token limit.
                batch = self.runtime.collator(
                    requests[offset : offset + 8], self.runtime.device
                )
                if tuple(batch["opt_mask"].shape[1:]) != (1, 2) or not bool(
                    batch["opt_mask"].all()
                ):
                    raise ValueError("Expected one complete binary question per row")
                logits = self.runtime.model(
                    *(
                        batch[k]
                        for k in (
                            "input_ids",
                            "attention_mask",
                            "opt_pos",
                            "opt_mask",
                            "q_pos",
                            "seg",
                        )
                    )
                ).float()
                policies.extend(
                    (logits / self.runtime.model.temperature)
                    .softmax(-1)[:, 0, :]
                    .tolist()
                )
        if len(policies) != len(tasks):
            raise ValueError("OpenJev question/response count mismatch")
        return decode_scores(
            tasks, {t.name: dict(zip(t.labels, p)) for t, p in zip(tasks, policies)}
        )
