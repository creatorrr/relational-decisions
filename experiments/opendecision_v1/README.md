# OpenDecision-Large development comparison

Evaluate `Tokz-labs/OpenDecision-Large` with the previously train-selected
`binary-reports-v2` grounding questions. The protocol was frozen before this
model's dev predictions were scored. No OpenDecision-specific prompt selection,
temperature fitting, or heldout evaluation is included.

## Model and input contract

- Model revision: `37ced3072592962b1e0cfdcc77eb511f8a3f7b54`.
- Inference package: [tokz-labs/OpenDecision](https://github.com/tokz-labs/OpenDecision)
  commit `fb5da5376cd72f699b001bb89be47b592fbce030`.
- 433,914,882 parameters; DeBERTa-v3-large cross-encoder, CPU FP32,
  four PyTorch threads, fixed seed, deterministic algorithms enabled.
- Same 48 dev programs / 24 worlds, 576 assessments and 272 queries;
  hard evidence, frontier batches of up to four propositions, search quantum 64.
- Same frozen questions, answer labels/descriptions, world prefix, and product
  mapping of positive/negative judgments to four evidence categories as the
  [GLiNER prompt comparison](../prompt_v2/README.md).

OpenDecision encodes each answer option in a separate bidirectional attention
row. A batch of four propositions produces eight binary questions and sixteen
encoder rows. Questions cannot attend to each other; the policy head's softmax
only normalizes answer scores within one question. This differs from GLiNER's
shared schema encoding. The model's candidate-local sigmoid `value` head is
unused. Policy temperature is 1; hard labels do not test probability calibration.

Its native decision header is capped at 20 tokens. Our frozen questions need
28–34 tokens, before any answer descriptions. To preserve them without changing
the checkpoint's token limits, each question goes into its own state:

```text
Question: {complete frozen question}

{unchanged world prefix}{world text}
```

The native decision name is `answer`. Choice options retain `yes` / `no` and
their original descriptions, serialized by OpenDecision as `name :: description`.
This is a transfer comparison with different native input layouts, not an
identical token serialization or a search for this model's best prompt.

The adapter checks all four truncation paths: decision header, candidate budget,
state budget, and the joint sequence position limit. Any truncation raises an
error. Dev states range from 196 to 287 tokens; answer options use 13–14 tokens,
so all requests fit. Actual native cross-encoder rows include the package's
additional wrapper tokens; the library's own serialization is unchanged.

## Reproduce

From the repository root, using Python 3.12 and a new environment:

```bash
uv venv .venv-opendecision --python 3.12
uv pip install --python .venv-opendecision/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r experiments/opendecision_v1/requirements.lock.txt
.venv-opendecision/bin/python experiments/opendecision_v1/run_dev.py \
  --output runs/opendecision-dev-reproduction
.venv-opendecision/bin/python experiments/opendecision_v1/probe.py \
  --output runs/opendecision-probe-reproduction.json
```

The lock preserves the previous comparison's installed dependencies and adds
only the pinned OpenDecision source. GLiNER is included for environment parity,
but this run never loads its models. The wrapper requires a fresh output and
cache directory, checks the frozen prior prompt selection, and records code
hashes, commands, timing, and Linux peak process RSS (including model loading).

The library adapter is also available through:

```bash
rd-bench --backend opendecision --split dev --output runs/opendecision-dev
```

Its defaults select the pinned model and `binary-reports-v2`. Model scores remain
uncalibrated. In soft mode, multiplying the two binary distributions introduces
an independence assumption; this experiment uses hard mode throughout.
