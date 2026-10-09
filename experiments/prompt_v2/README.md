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

## Dev results

Both columns use the frozen `binary-reports-v2` prompt and matched dependencies:

| Metric | Decide 340M | Decide 1B |
| --- | ---: | ---: |
| Assessment accuracy | 26.56% (153/576) | 26.74% (154/576) |
| Assessment macro-F1 | 0.1938 | 0.2533 |
| Query probability MAE | 0.1732 | 0.1697 |
| Query probabilities within 1e-9 | 59.56% (162/272) | 63.60% (173/272) |
| Paraphrase assessment agreement | 73.61% | 43.06% |

The larger checkpoint gets only one additional assessment correct. An
exploratory paired bootstrap over the 24 worlds gives a 95% percentile interval
of -3.47 to +3.65 percentage points for the accuracy difference (1B minus 340M;
5,000 draws, seed 20261009). This provides no clear evidence of an accuracy
benefit on this development set. The 1B model has better macro-F1 and query
metrics, but both remain below the 32.64% always-supported accuracy baseline.

Their error patterns differ substantially. The 340M model predicts unknown
462 times, supported 37, refuted 77, and both zero times; it misses every
contradictory evidence category. The 1B model predicts both 311 times, supported
95, refuted 71, and unknown 99 times. It treats 107 of the 188 positive-only
cases as conflicting reports. More varied predictions do not yet amount to
reliable grounding.

The winner's 32.64% train-panel accuracy falls to 26.56% on 340M dev. The screen
and dev also differ in wording and batch scheduling, so this cannot isolate
which shift caused the drop. The prompt was selected with 340M, then transferred
unchanged to 1B; these results do not estimate either model's best achievable
performance after further prompt search or training.

For historical context, the original-prompt 340M run scored 23.61% accuracy,
0.1707 macro-F1, and 0.1985 query MAE. That run used older dependencies. The
matched train screen here isolates formulation changes; comparing the historical
dev run with this experiment also changes the runtime.

The independently enumerating reference reproduced all 272 submitted query
probabilities for each model from its predicted assessments. Both archived
metric files were rescored, and model settings and runtime source hashes match
apart from checkpoint identity. `comparison.json` records the audit and paired
comparison; `summarize.py` reproduces it from the checked-in dev artifacts.
All 19 unit tests and the independent ProbLog benchmark validation passed.
The evaluated runtime source is commit `19bbbe552e584391c06fad630a9b966607476cba`.

### Execution and recovery

The 1B run made 161 uncached backend calls in 1,031.3 seconds after initialization
(17.2 minutes). Peak process RSS including loading was 9,744,812 KiB (9.29 GiB).
Total wall time including initialization was 1,075.8 seconds. It fits the
16 GiB box in FP32; no quantization was used.

An environment restart killed the 340M process after 39 complete programs and
122 completed model calls. Its cache survived. A new process replayed those
programs and computed the remaining nine, making 27 new calls. All 39 replayed
prediction records exactly match the originals; the final run covers all 48
programs and 149 batches. No prompts, weights, or settings changed on resume.

The resumed metadata's 113.2-second elapsed time is **not a fresh full-run
timing**. The original 39 completed programs recorded 459.1 seconds, but the
restart prevents reporting one uninterrupted wall time or a combined peak RSS.
The resumed process alone peaked at 4,327,020 KiB. These are single-run resource
observations, not a controlled throughput comparison.

Files with the `decide-340m-dev.interrupted` prefix preserve the original
partial run; the primary `decide-340m-dev` files contain the complete resumed
result. Resources record the original hashes, interruption, and cache reuse.
The protocol's fresh-cache timing condition was therefore interrupted for
340M; its accuracy comparison is still complete and uses the frozen protocol.
Heldout contents remained unused and the sealed archive checksum is unchanged.

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
# Audit the already archived dev results (standard library only):
.venv/bin/python experiments/prompt_v2/summarize.py
```

Run the two dev commands sequentially on a machine with this memory budget.
Each dev directory contains predictions, traces, metrics, metadata, and Linux
process resource measurements. The wrapper requires fresh output and cache
directories and records the protocol, selection, and runtime source hashes.
Model weights and caches are not committed.
