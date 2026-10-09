"""Optional CPU GLiNER adapter. Imports ML dependencies only when constructed."""

import json
import platform
import subprocess
from importlib.metadata import distribution, version
from pathlib import Path

from .decisions import LABELS

DEFAULT_MODEL = "fastino/gliner2.5-small-v1"
DEFAULT_REVISION = "df5910e44bc4ffdb0d95399a83b0ca4516349aa5"
PROMPT_VERSION = "explicit-reports-v1"
LABEL_DESCRIPTIONS = {
    "supported": "Only the positive assertion is explicitly reported.",
    "refuted": "Only the negative assertion is explicitly reported.",
    "both": "Both assertions are explicitly reported, so reports conflict.",
    "unknown": "Neither assertion is explicitly reported; information is missing.",
}
WORLD_PREFIX = "Unordered reports about one snapshot. No report has priority. Missing information is unknown.\n\n"


class GLiNERBackend:
    def __init__(
        self, model=DEFAULT_MODEL, revision=None, *, threads=4, max_tokens=4096
    ):
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
            "prompt_version": PROMPT_VERSION,
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

    @property
    def identity(self):
        return self._identity

    def assess(self, world, candidates):
        from gliner2.classification import ClassificationSchema
        from gliner2.classification.compiler import compile_schema

        schema = ClassificationSchema()
        task_ids = {}
        for i, candidate in enumerate(candidates):
            task = f"decision_{i}"
            task_ids[task] = candidate.id
            instruction = (
                f"Assess explicit reports. Positive assertion: {candidate.proposition}. "
                f"Negative assertion: {candidate.negative_proposition}."
            )
            schema.single(
                task, LABEL_DESCRIPTIONS, instruction=instruction, activation="softmax"
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
        if len(batch.schema_tokens_list[0]) != len(candidates):
            raise ValueError(
                "GLiNER preprocessing did not preserve all requested tasks"
            )
        scores = self.classifier.score(text, compiled)
        return {
            cid: {label: scores.probability(task, label) for label in LABELS}
            for task, cid in task_ids.items()
        }
