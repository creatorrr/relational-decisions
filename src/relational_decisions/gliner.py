"""Optional CPU GLiNER adapter. Imports ML dependencies only when constructed."""

import json
import platform
import subprocess
from dataclasses import asdict
from importlib.metadata import distribution, version
from pathlib import Path

from .decisions import LABELS, Candidate
from .prompts import (
    DEFAULT_PROMPT,
    LABEL_DESCRIPTIONS,
    PROMPTS,
    WORLD_PREFIX,
    build_tasks,
    decode_scores,
)

DEFAULT_MODEL = "fastino/gliner2.5-small-v1"
DEFAULT_REVISION = "df5910e44bc4ffdb0d95399a83b0ca4516349aa5"
PROMPT_VERSION = DEFAULT_PROMPT


def check_encoder_runtime(config, transformers_version):
    encoder = config.get("encoder_config", {})
    # The old 4.x ModernBERT implementation silently ignores this nested RoPE
    # configuration and applies different local-attention rotary frequencies.
    if encoder.get("model_type") == "modernbert" and encoder.get("rope_parameters"):
        release = tuple(int(x) for x in transformers_version.split(".")[:2])
        if release < (5, 17):
            raise ValueError(
                "This ModernBERT checkpoint requires the tested Transformers 5.17+ "
                "runtime for its rope_parameters configuration; use the prompt_v2 lock file."
            )


class GLiNERBackend:
    def __init__(
        self,
        model=DEFAULT_MODEL,
        revision=None,
        *,
        threads=4,
        max_tokens=4096,
        prompt=DEFAULT_PROMPT,
    ):
        if prompt not in PROMPTS:
            raise ValueError(f"Unknown grounding prompt: {prompt}")
        import gliner2
        import torch
        from gliner2 import AutoExtractor
        from gliner2.classification import Classifier
        from huggingface_hub import HfApi, snapshot_download

        if threads < 1 or max_tokens < 1:
            raise ValueError("Thread count and token limit must be positive")
        if revision is None:
            revision = (
                DEFAULT_REVISION
                if model == DEFAULT_MODEL
                else HfApi().model_info(model).sha
            )
        # Resolve named revisions once; cache identity always uses an immutable SHA.
        if len(revision) != 40 or any(
            c not in "0123456789abcdef" for c in revision.lower()
        ):
            revision = HfApi().model_info(model, revision=revision).sha
        path = snapshot_download(
            model,
            revision=revision,
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
        )
        check_encoder_runtime(
            json.loads((Path(path) / "config.json").read_text()),
            version("transformers"),
        )
        torch.set_num_threads(threads)
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        self.model = AutoExtractor.from_pretrained(path, map_location="cpu")
        self.model.float().eval()
        self.classifier = Classifier(self.model).eval()
        self.max_tokens = max_tokens
        package_source = None
        direct = distribution("gliner2").read_text("direct_url.json")
        if direct:
            info = json.loads(direct)
            package_source = info.get("vcs_info", {}).get("commit_id")
        checkout = Path(gliner2.__file__).resolve().parent.parent
        if package_source is None and (checkout / ".git").exists():
            package_source = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=checkout, text=True
            ).strip()
        self._identity = {
            "backend": "gliner",
            "model": model,
            "revision": revision,
            "prompt_version": prompt,
            "world_prefix": WORLD_PREFIX,
            "label_order": LABELS,
            "label_descriptions": LABEL_DESCRIPTIONS,
            "dtype": "float32",
            "device": "cpu",
            "threads": threads,
            "max_tokens": max_tokens,
            "python": platform.python_version(),
            "machine": platform.machine(),
            "gliner_source_commit": package_source,
            "packages": {
                p: version(p)
                for p in ("gliner2", "torch", "transformers", "tokenizers")
            },
            "deterministic_algorithms": True,
            "activation": "softmax",
            "calibration": "none",
        }
        self.set_prompt(prompt)

    def set_prompt(self, prompt):
        if prompt not in PROMPTS:
            raise ValueError(f"Unknown grounding prompt: {prompt}")
        self.prompt = prompt
        self._identity["prompt_version"] = prompt
        self._identity.pop("task_templates", None)
        self._identity.pop("score_composition", None)
        if prompt != DEFAULT_PROMPT:
            template = Candidate(
                "candidate", ("p",), ("not_p",), "{positive}", "{negative}"
            )
            self._identity["task_templates"] = [
                asdict(t) for t in build_tasks([template], prompt)
            ]
            self._identity["score_composition"] = (
                "independent-positive-negative-product"
                if prompt == "binary-reports-v2"
                else "four-way-categorical"
            )

    @property
    def identity(self):
        return self._identity

    def assess(self, world, candidates):
        from gliner2.classification import ClassificationSchema
        from gliner2.classification.compiler import compile_schema

        schema = ClassificationSchema()
        tasks = build_tasks(candidates, self.prompt)
        for task in tasks:
            schema.single(
                task.name,
                task.labels,
                instruction=task.instruction,
                activation="softmax",
            )
        # Bypass the facade's order-insensitive compilation cache.
        compiled = compile_schema(schema)
        text = WORLD_PREFIX + world
        # Inspect the actual encoded sequence. Never silently truncate evidence
        # or silently split a joint schema, because either changes the assessment.
        batch = self.model.processor.collate_fn_inference(
            [(text, compiled.build())],
            architecture=self.model.architecture,
        )
        encoded_length = int(batch.attention_mask[0].sum())
        if encoded_length > self.max_tokens:
            raise ValueError(
                f"Joint input has {encoded_length} tokens, above limit {self.max_tokens}; reduce the declared batch size"
            )
        if len(batch.schema_tokens_list[0]) != len(tasks):
            raise ValueError(
                "GLiNER preprocessing did not preserve all requested tasks"
            )
        scores = self.classifier.score(text, compiled)
        return decode_scores(
            tasks,
            {
                task.name: {
                    label: scores.probability(task.name, label) for label in task.labels
                }
                for task in tasks
            },
        )
