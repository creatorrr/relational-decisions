"""H2O-Lightning's native text decisions, evaluated locally on a BF16 CPU."""

import hashlib
import importlib.util
import json
import platform
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from .decisions import LABELS, Candidate
from .prompts import WORLD_PREFIX, build_tasks, decode_scores

DEFAULT_MODEL = "h2oai/h2o-lightning-4b"
DEFAULT_REVISION = "672dc01ed37a516357cd3c7da777c96699d3f16c"
PROMPT = "binary-reports-v2"


def label_token_ids(tokenizer, text, labels):
    """Match the publisher's answer-slot check, including the leading space."""
    base = tokenizer.encode(text, add_special_tokens=False)
    ids = []
    for label in labels:
        full = tokenizer.encode(text + " " + label, add_special_tokens=False)
        if len(full) != len(base) + 1 or full[:-1] != base:
            raise ValueError(f"Answer label {label!r} is not one token at this slot")
        ids.append(full[-1])
    if len(set(ids)) != len(ids):
        raise ValueError("Answer labels must have distinct token IDs")
    return base, ids


class H2OLightningBackend:
    def __init__(self, *, threads=4):
        if threads < 1:
            raise ValueError("Thread count must be positive")
        import torch
        import transformers.models.qwen3_5.modeling_qwen3_5 as qwen
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

        path = Path(
            snapshot_download(
                DEFAULT_MODEL,
                revision=DEFAULT_REVISION,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "tokenizer*.json",
                    "chat_template.jinja",
                    "generation_config.json",
                    "h2o_lightning_shim.py",
                    "serve_config.json",
                ],
            )
        )
        # Only use the reviewed shim's pure rendering/scoring functions. No server.
        spec = importlib.util.spec_from_file_location(
            "_relational_h2o_" + DEFAULT_REVISION, path / "h2o_lightning_shim.py"
        )
        self.shim = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.shim)
        self.contract = self.shim.Contract(
            json.loads((path / "serve_config.json").read_text()), env={}
        )
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        # Use Transformers' PyTorch fallback, avoiding optional GPU kernel wrappers.
        unwrapped = []
        for name in (
            "causal_conv1d_fn",
            "torch_chunk_gated_delta_rule",
            "torch_recurrent_gated_delta_rule",
        ):
            function = getattr(qwen, name)
            if hasattr(function, "__wrapped__"):
                setattr(qwen, name, function.__wrapped__)
                unwrapped.append(name)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).eval()
        if next(self.model.parameters()).device.type != "cpu":
            raise ValueError("This experimental backend requires a CPU")
        template = Candidate(
            "candidate", ("p",), ("not_p",), "{positive}", "{negative}"
        )
        self._identity = {
            "backend": "h2o-lightning",
            "adapter_version": "native-choice-cpu-v1",
            "model": DEFAULT_MODEL,
            "revision": DEFAULT_REVISION,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
            "prompt_version": PROMPT,
            "world_prefix": WORLD_PREFIX,
            "task_templates": [asdict(t) for t in build_tasks([template], PROMPT)],
            "label_order": LABELS,
            "score_composition": "independent-positive-negative-product",
            "question_type": "choice",
            "questions_per_attention_sequence": 1,
            "question_batch_size": 1,
            "attention": "causal Qwen3.5: 24 Gated DeltaNet and 8 full-attention layers",
            "activation": "softmax over answer-label logits",
            "temperature": self.contract.temperature,
            "calibration": "published choice temperature; no benchmark calibration",
            "noul_floor_applied": False,
            "serve_config": self.contract.cfg,
            "source_sha256": {
                name: hashlib.sha256((path / name).read_bytes()).hexdigest()
                for name in (
                    "h2o_lightning_shim.py",
                    "serve_config.json",
                    "config.json",
                )
            },
            "dtype": "bfloat16 backbone; float32 selected LM-head rows and logits",
            "readout": "last hidden state projected onto verified answer-label rows",
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
            "cpu_unwrapped_functions": unwrapped,
            "deterministic_algorithms": True,
            "truncation": "reject",
            "use_cache": False,
        }

    @property
    def identity(self):
        return self._identity

    def requests(self, world, candidates):
        tasks = build_tasks(candidates, PROMPT)
        state = WORLD_PREFIX + world
        if (
            len(self.tokenizer.encode(state, add_special_tokens=False))
            > self.contract.max_state_tokens
        ):
            raise ValueError("World exceeds the published state limit")
        requests = []
        for task in tasks:
            question = {
                "type": "choice",
                "instructions": task.instruction,
                "criteria": task.labels,
            }
            text, names, _, labels = self.contract.question_prompt(state, question)
            tokens, ids = label_token_ids(self.tokenizer, text, labels)
            if len(tokens) + 1 > 40960:
                raise ValueError("Prompt exceeds the published serving context limit")
            if names != list(task.labels):
                raise ValueError("Native option order differs from the frozen task")
            requests.append((text, tokens, ids))
        return tasks, requests

    def hidden_and_logits(self, tokens, ids):
        import torch

        with torch.inference_mode():
            hidden = (
                self.model.model(input_ids=torch.tensor([tokens]), use_cache=False)
                .last_hidden_state[:, -1, :]
                .float()
            )
            # The publisher requires FP32 head computation. Selecting the relevant
            # rows first avoids an unnecessary full-vocabulary FP32 head copy.
            weight = self.model.lm_head.weight[ids].float()
            logits = torch.nn.functional.linear(hidden, weight)[0]
        return hidden, logits

    def assess(self, world, candidates):
        tasks, requests = self.requests(world, candidates)
        scores = {}
        for task, (_, tokens, ids) in zip(tasks, requests):
            _, logits = self.hidden_and_logits(tokens, ids)
            # Subtracting full-vocabulary logsumexp cancels in label softmax.
            probabilities = self.shim.probabilities(
                logits.tolist(), self.contract.temperature
            )
            scores[task.name] = dict(zip(task.labels, probabilities))
        return decode_scores(tasks, scores)
