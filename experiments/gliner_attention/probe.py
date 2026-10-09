"""Reproducible local GLiNER attention and schema interaction probe."""
import argparse
import json
import platform
import subprocess
import time
from importlib.metadata import distribution, version
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from gliner2 import AutoExtractor
from gliner2.classification import Classifier, ClassificationSchema
from gliner2.classification.compiler import compile_schema

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("runs/gliner_attention/results.json"))
args = parser.parse_args()
MODEL_ID = "fastino/gliner2.5-small-v1"
MODEL_REVISION = "df5910e44bc4ffdb0d95399a83b0ca4516349aa5"
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

def emit(label, value):
    print(label, json.dumps(value, default=str), flush=True)

revision = MODEL_REVISION
emit("downloading", {"model": MODEL_ID, "revision": revision})
model_path = snapshot_download(
    MODEL_ID, revision=revision,
    allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"],
)
start = time.perf_counter()
model = AutoExtractor.from_pretrained(model_path, map_location="cpu")
model.float().eval()
clf = Classifier(model).eval()

def source_commit():
    """Record package provenance without depending on a sibling source clone."""
    direct_url = distribution("gliner2").read_text("direct_url.json")
    if direct_url:
        info = json.loads(direct_url)
        commit = info.get("vcs_info", {}).get("commit_id")
        if commit:
            return commit
    import gliner2
    checkout = Path(gliner2.__file__).resolve().parent.parent
    if (checkout / ".git").exists():
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
    return None

meta = {
    "model": MODEL_ID, "revision": revision,
    "source_commit": source_commit(),
    "python": platform.python_version(),
    "packages": {p: version(p) for p in ["gliner2", "torch", "transformers"]},
    "device": str(next(model.parameters()).device),
    "dtype": str(next(model.parameters()).dtype),
    "parameters": sum(p.numel() for p in model.parameters()),
    "load_seconds": time.perf_counter() - start,
    "encoder": type(model.encoder).__name__,
    "classifier": str(model.classifier),
}
emit("loaded", meta)

captured = {}
def capture_encoder(module, args, kwargs):
    captured["inputs"] = {
        k: v.detach().clone() for k, v in kwargs.items() if torch.is_tensor(v)
    }

handle = model.encoder.register_forward_pre_hook(capture_encoder, with_kwargs=True)

text = "Battery dies before lunch, but the keyboard and the screen are the best I have used on a laptop."
tasks = {
    "sentiment": ["positive", "negative", "mixed", "neutral"],
    "battery": ["good", "bad", "not_mentioned"],
    "keyboard": ["good", "bad", "not_mentioned"],
    "screen": ["good", "bad", "not_mentioned"],
}

def make_schema(order, overrides=None):
    schema = ClassificationSchema()
    for task in order:
        schema.single(task, (overrides or {}).get(task, tasks[task]))
    return schema

def score_case(name, schema, input_text=text):
    start = time.perf_counter()
    # Compile explicitly: the facade's order-insensitive cache fingerprint can
    # otherwise reuse the first task order and invalidate the reversal probe.
    scores = clf.score(input_text, compile_schema(schema))
    result = {
        "seconds": time.perf_counter() - start,
        "tokens": captured["inputs"]["input_ids"].shape[1],
        "logits": {task: dict(values) for task, values in scores.tasks.items()},
        "probabilities": {
            task: {label: scores.probability(task, label) for label in values}
            for task, values in scores.tasks.items()
        },
    }
    emit(name, result)
    return result

cases = {}
cases["sentiment_alone"] = score_case("sentiment_alone", make_schema(["sentiment"]))
combined = make_schema(list(tasks))
cases["combined"] = score_case("combined", combined)
combined_inputs = {k: v.clone() for k, v in captured["inputs"].items()}
cases["combined_repeat"] = score_case("combined_repeat", combined)
cases["reversed_tasks"] = score_case("reversed_tasks", make_schema(list(reversed(tasks))))
cases["changed_later_labels"] = score_case(
    "changed_later_labels", make_schema(list(tasks), {"screen": ["red", "blue", "green"]})
)
cases["swapped_later_labels"] = score_case(
    "swapped_later_labels", make_schema(list(tasks), {"screen": ["bad", "good", "not_mentioned"]})
)

# Distinguish one joint schema from separate examples in the batch dimension.
separate_scores = clf.batch_score(
    [text] * len(tasks),
    [compile_schema(make_schema([task])) for task in tasks],
)
separate_batch = {
    task: dict(scores.tasks[task]) for task, scores in zip(tasks, separate_scores)
}
emit("separate_batch_logits", separate_batch)
public_api = model.classify_text(text, tasks, include_confidence=True)
emit("public_classify_text", public_api)
handle.remove()

# Inspect the actual per-layer masks and attention weights on one joint input.
mask_records = []
def capture_attention(module, args, kwargs):
    mask = kwargs.get("attention_mask")
    if mask is None and len(args) > 1:
        mask = args[1]
    if mask is not None:
        mask_records.append({
            "module": type(module).__name__, "shape": list(mask.shape),
            "dtype": str(mask.dtype), "unique": mask.unique().tolist(),
        })

attention_modules = [
    (name, module) for name, module in model.encoder.named_modules()
    if name.endswith("attention.self")
]
emit("attention_modules", [name for name, _ in attention_modules])
hooks = [m.register_forward_pre_hook(capture_attention, with_kwargs=True)
         for _, m in attention_modules]
with torch.inference_mode():
    output = model.encoder(**combined_inputs, output_attentions=True, return_dict=True)
for h in hooks:
    h.remove()

tokens = model.processor.tokenizer.convert_ids_to_tokens(combined_inputs["input_ids"][0].tolist())
emit("tokens", list(enumerate(tokens)))
compiled = clf.compile_schema(combined)
batch = model.processor.collate_fn_inference(
    [(text, compiled.build())], architecture=model.architecture,
)
groups = batch.schema_special_indices[0]
emit("schema_markers", groups)
attention = {"layer_masks": mask_records, "layers": []}
for layer_index, weights in enumerate(output.attentions):
    w = weights[0]
    length = w.shape[-1]
    upper = torch.triu(torch.ones(length, length, dtype=torch.bool), diagonal=1)
    first_slot = groups[0][1]
    later_slot = groups[-1][1]
    attention["layers"].append({
        "layer": layer_index,
        "future_attention_mass_mean": w[:, upper].sum().item() / (w.shape[0] * length),
        "first_slot_to_later_slot_mean": w[:, first_slot, later_slot].mean().item(),
        "later_slot_to_first_slot_mean": w[:, later_slot, first_slot].mean().item(),
    })
emit("attention", attention)

# Hold positions and sequence length fixed, changing only the final task's label
# tokens. This separates semantic cross-slot influence from position shifts.
variant_schema = make_schema(list(tasks), {"screen": ["bad", "good", "not_mentioned"]})
variant_batch = model.processor.collate_fn_inference(
    [(text, clf.compile_schema(variant_schema).build())], architecture=model.architecture,
)
same_shape = batch.input_ids.shape == variant_batch.input_ids.shape
perturbation = {"same_shape": same_shape}
if same_shape:
    changed = (batch.input_ids[0] != variant_batch.input_ids[0]).nonzero().flatten()
    perturbation["changed_positions"] = changed.tolist()
    with torch.inference_mode():
        other = model.encoder(
            input_ids=variant_batch.input_ids,
            attention_mask=variant_batch.attention_mask,
            return_dict=True,
        ).last_hidden_state
    first_positions = groups[0][1:]
    perturbation["first_task_hidden_max_abs_delta"] = (
        output.last_hidden_state[0, first_positions] - other[0, first_positions]
    ).abs().max().item()
    with torch.inference_mode():
        before_logits = model.classifier(output.last_hidden_state[0, first_positions]).squeeze(-1)
        after_logits = model.classifier(other[0, first_positions]).squeeze(-1)
    perturbation["first_task_logits_before"] = before_logits.tolist()
    perturbation["first_task_logits_after"] = after_logits.tolist()
    perturbation["first_task_logit_max_abs_delta"] = (before_logits - after_logits).abs().max().item()
emit("perturbation", perturbation)

result = {
    "metadata": meta, "text": text, "cases": cases,
    "tokens": tokens, "schema_markers": groups,
    "attention": attention, "perturbation": perturbation,
    "separate_batch_logits": separate_batch, "public_classify_text": public_api,
}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
emit("saved", str(args.output.resolve()))
