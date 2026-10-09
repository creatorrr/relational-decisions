# Prompt selection and the Decide 1B comparison

This experiment changes the grounding formulation before comparing two model
checkpoints. The engine and its exact probabilistic rules are unchanged.
Heldout contents were not inspected or evaluated.

## Protocol

`protocol.json` was written before the train screen. It fixes 12 train programs
from 12 distinct worlds: one per reasoning family, four per everyday domain,
and six of each text rendering. The selection rule uses family/domain and
lexicographic IDs, not gold labels or model results. Each program contributes
all 12 candidate assessments, for 144 assessments total.

Four formulations were evaluated with the pinned Decide 340M checkpoint.
The screen uses fixed groups of four candidates in ID order, with no cache.
Highest macro-F1 selects the prompt, with accuracy and then protocol order
breaking ties. The selected prompt was frozen before evaluating it on all
48 dev renderings (24 worlds, 576 assessments, 272 queries) with both models.
This is a small train selection experiment and a development comparison,
not a final generalization result on heldout.

| Prompt | Train accuracy | Train macro-F1 |
| --- | ---: | ---: |
| Original: `explicit-reports-v1` | 23.61% | 0.1216 |
| Simpler labels: `direct-status-v2` | 25.00% | 0.1923 |
| Proposition-specific options: `concrete-options-v2` | 29.86% | 0.2340 |
| Separate report questions: `binary-reports-v2` | **32.64%** | **0.2930** |

The majority label in this train panel is supported (46/144, 31.94%). Even the
winning prompt is only slightly above that accuracy baseline. Full predictions,
distributions, confusion matrices, and selection metadata are in `train-screen/`.
The original prompt and defaults remain available; selection does not rewrite
the old experiments.

## Selected formulation

For each proposition, ask separately whether the reports explicitly state:

1. The positive statement, such as "Room Oak needs maintenance".
2. Its explicit negative, such as "Room Oak does not need maintenance".

Both questions instruct the model to match the exact named entity and accept
paraphrases. Each has `yes` and `no` labels. Four candidates therefore produce
eight classifier tasks in one jointly encoded schema. The same original world
text and prefix are supplied; gold labels and symbolic query answers are never
passed to the model.

The two distributions are combined by multiplication into the four evidence
categories. This assumes independence between positive and negative report
judgments; it is not a calibrated joint model. These dev runs use hard mode,
which selects one category as observed evidence. The program's declared random
choices still retain their exact probabilities. See [engine semantics](../../docs/engine.md).

## Runtime and reproducibility

Both new dev runs use CPU FP32, four PyTorch threads, deterministic algorithms,
the same dependency lock, and the same frontier settings: four candidates per
batch, 64 agenda events between service opportunities, and complete assessment
coverage. Different model predictions may produce different frontier batches.
The train screen's fixed groups also differ from the dev frontier policy.

- Decide 340M: `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90`.
- Decide 1B: `67af949e17bebd20f05fe020a971079889e7a462`.
- GLiNER source: `08c8c8df0803075035ac6a02e8ee8c5554d2e06b`.
- Transformers: `5.17.0`; PyTorch: `2.14.1+cpu`.

The 1B checkpoint contains 1,188,796,693 saved FP32 tensor elements. It uses
ModernBERT/Ettin with 28 layers, a 1,792-wide hidden state, full attention every
third layer, and local attention between those layers. Its saved configuration
sets causal masking off. This compares different encoder architectures as
well as parameter counts, so it cannot isolate the effect of size alone.

The newer runtime preserves the 1B checkpoint's nested rotary configuration;
the original Transformers 4.57 runtime would use different local frequencies.
The adapter now rejects that combination. Historical dev results in
`engine_v1` used Transformers 4.57.6 and tokenizers 0.22.2, whereas this experiment
uses Transformers 5.17.0 and tokenizers 0.23.2. Historical comparisons therefore
also differ in dependencies; the two new model runs use matched dependencies.

From the repository root, use a dedicated virtual environment:

```bash
python -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r experiments/prompt_v2/requirements.lock.txt
.venv/bin/python experiments/prompt_v2/run_screen.py --output runs/new-screen
.venv/bin/python experiments/prompt_v2/run_dev.py --model 1b \
  --selection runs/new-screen/selection.json --output runs/new-1b-dev
.venv/bin/python experiments/prompt_v2/run_dev.py --model 340m \
  --selection runs/new-screen/selection.json --output runs/new-340m-dev
```

Run the two dev commands sequentially on a machine with this memory budget.
Each dev directory contains predictions, traces, metrics, metadata, and Linux
process resource measurements. The wrapper requires fresh output and cache
directories and records the protocol, selection, and runtime source hashes.
Model weights and caches are not committed.
