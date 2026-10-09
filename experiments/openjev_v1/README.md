# Independent OpenJev development comparison

Evaluate [`com-kotobalabs/open-jev-deberta-v3-large`](https://huggingface.co/com-kotobalabs/open-jev-deberta-v3-large)
at revision `188ee67a5c93122b916e5acd5bdb0cb3623e380a`. This is an independent
implementation inspired by Jev's interface, not TypeSafe's hosted Jev model.
The model snapshot includes the reviewed inference source; code and weights
are pinned together. No API credentials are used.

The frozen `binary-reports-v2` questions, world prefix, yes/no descriptions,
and hard-mode mapping are reused without model-specific prompt selection.
The original plan covered all 48 dev inputs; after the user requested faster
practical iteration, the actual evaluation was narrowed to the first eight,
matching the shortened GLiNER layout check. Heldout remains sealed.

## Quick result

On eight programs (96 assessments, 42 queries), OpenJev reaches **39.58%**
assessment accuracy, **0.3395** macro-F1, and **0.1170** query-probability MAE.
OpenDecision reaches 43.75%, 0.3846, and 0.1195 on exactly those same inputs.
OpenJev matches the subset's majority-label accuracy; this small check is not
a reliable ranking or a heldout result.

The model predicts supported/refuted/both/unknown 30/12/50/4 times. It identifies
14/18 contradictory cases but only 2/20 unknown and 4/20 refuted cases. Missing
information and explicit negatives remain weak points. All 42 query outputs
match the exact reference given its predicted facts.

The checkpoint has 437,159,937 parameters. The eight-program CPU FP32 run took
186.26 seconds including initialization and peaked at 2.46 GiB RSS. The probe
confirms bidirectional attention within each row, identical repeated batched
outputs, and agreement with the public API within 2.24e-7 in four-category
weights, with identical hard decisions.

See the [matched comparison and practical notes](../question_isolation/README.md).
Artifacts are saved as `openjev-quick8.*` and `probe.json`; no full 48-program
OpenJev result is claimed.

## Input and batching

OpenJev is a bidirectional DeBERTa-v3-large encoder that natively places state
and questions in one sequence. Its span-pooling head combines each question's
representation with each option's representation, then applies a softmax over
that question's options. Its native architecture can share attention between
questions; this experiment explicitly isolates them.

The state cap is 256 tokens and the entire sequence cap is 512. Eight full
questions with our answer descriptions exceed that total limit. We therefore
use **one question per attention sequence**, matching the GLiNER isolation
control, and batch up to eight independent rows for CPU efficiency. A native
`choice` carries the complete frozen instructions; its two option strings are
`yes :: {description}` and `no :: {description}`. This preserves the original
label order and descriptions rather than switching to the shorter `noul` API.

The adapter uses the published collator and model directly to batch independent
rows; the publisher's `decide_batch` currently loops over single calls. The
probe verifies our batched scores against the public `decide` method. The
collator's required training-label field receives a constant placeholder;
that tensor is never passed to inference and no benchmark gold is loaded.

World-state truncation is rejected before collation. The native collator raises
on sequences above 512 tokens. No token limits are enlarged. The published
temperature of 1.05 is retained; it was not fitted on this benchmark and is not
evidence of calibration here. Positive temperature does not change binary
argmax decisions in this hard-mode experiment.

Across dev, states use 163–249 tokens and complete isolated rows use 219–310.
Groups of eight consecutive questions would need 602–702 tokens, exceeding
the native 512-token limit.

## Reproduce

Use the Python 3.12 environment pinned in
[`../opendecision_v1/requirements.lock.txt`](../opendecision_v1/requirements.lock.txt).
OpenJev requires no additional dependency packages. From the repository root:

```bash
python experiments/openjev_v1/probe.py --output runs/openjev-probe.json
python experiments/openjev_v1/run_dev.py --output runs/openjev-dev
# Equivalent benchmark entrypoint:
rd-bench --backend openjev --split dev --output runs/openjev-dev-direct
```

The wrapper requires fresh output/cache directories and records commands,
protocol and source hashes, timing, and peak process RSS. The adapter records
hashes of all bundled inference source files as well as the model revision,
runtime versions, CPU FP32 precision, and deterministic settings.

The commands above reproduce the original full-run plan. For the actual
eight-program diagnostic, run:

```bash
python experiments/question_isolation/run_quick.py --model openjev \
  --output runs/openjev-quick8
```
