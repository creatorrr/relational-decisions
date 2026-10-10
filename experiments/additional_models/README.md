# Additional native decision-model trials

This diagnostic compares IFM K2-Type-0.9B and Decision-2.0-Lux-9B Q4_K_M
with the saved H2O Q8_0 results on the same first eight public dev programs.
The subset contains 96 proposition assessments and 42 queries. It is a small
capability check, not a heldout evaluation or a model-selection sweep.

`protocol.json` was saved before benchmark predictions. The logical questions,
positive/negative report formulation, original candidate request groups, label
mapping, and exact evaluator are unchanged. Each native adapter splits its
binary tasks into groups of at most eight. No prompt, threshold, option order,
or temperature is tuned against the benchmark. Public dev gold is read only
once all eight predictions for that model have been produced. Heldout data
is not opened. H2O is **not rerun**, as requested.

## Matched dev results

| Model | Correct assessments | Macro-F1 | Exact queries | Query MAE | Wall time | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2O Q8_0, saved baseline | 85/96 (88.54%) | 0.8403 | 40/42 | 0.01467 | 8.89 min, historical | 4.61 GiB server + 0.49 GiB client |
| K2-Type-0.9B, native BF16 CPU | 32/96 (33.33%) | 0.2922 | 32/42 | 0.08333 | 4.21 min | 2.16 GiB process |
| Lux-9B Q4_K_M, native Decision2 | 80/96 (83.33%) | 0.7544 | 41/42 | 0.00514 | 58.44 min | 8.82 GiB server + 0.021 GiB client |

**H2O remains the strongest grounding model on this diagnostic.** Lux answers
one more downstream query exactly and has lower query MAE, while making sixteen
primitive assessment errors rather than eleven. Query performance can conceal
primitive errors when affected facts do not influence a query's answer.

| True evidence state | H2O correct | K2 correct | Lux correct |
| --- | ---: | ---: | ---: |
| supported | 38/38 | 13/38 | 37/38 |
| refuted | 20/20 | 8/20 | 20/20 |
| both | 7/18 | 11/18 | 3/18 |
| unknown | 20/20 | 0/20 | 20/20 |

K2 overpredicts `both`: 61 of 96 assessments, including all fourteen of its
unknown-to-both mistakes. It predicts no `unknown` assessments. Its higher recall
on true contradictions comes with many false contradictions. Lux collapses
fifteen of eighteen contradictory reports to one side, accounting for fifteen
of its sixteen total grounding errors. These are results under the frozen
explicit-report formulation, not claims about every task the models support.

The two new runs are sequential and use the same four-thread CPU allocation.
Historical H2O timings are shown for context only. Different model sizes,
quantizations, native renderers, numerical paths, and runtime implementations
prevent interpreting these as a quantization ablation or controlled speedup.
Server peak RSS is the kernel's lifetime `VmHWM`; client/process peak RSS comes
from `getrusage`. Separate peaks should not be added as simultaneous memory use.

[Independent verification](independent-verification.json) confirms all three
runs: complete 96-assessment/42-query coverage, normalized distributions,
hard-label consistency, original replayed candidate groups, recorded metric
agreement, and all 42 query probabilities per model against the independent
exact solver given its predicted facts. No decision backend is promoted by
this experiment. Machine-readable results are in [comparison.json](comparison.json).

```bash
python experiments/additional_models/verify_results.py \
  experiments/gguf_v1/h2o-q8-quick8 \
  experiments/additional_models/k2-quick8 \
  experiments/additional_models/lux-q4-quick8
```

## Native interfaces

K2 uses the pinned publisher's `jev/model.py` and `jev/encode.py`: a BF16
K2-Horizon backbone, float32 pointer head, published temperature
1.4778441535263076, block-causal attention, and per-question position resets.
The language-model output head is never used. The CPU adaptation changes device
placement from the publisher's CUDA server and retains unrounded native
probabilities instead of the HTTP server's four-decimal output. State, option,
and total sequence limits are checked explicitly; the publisher encoder's
silent truncation/skipping behavior is rejected.

Lux uses the explicitly requested **Q4_K_M** file, checksum verified, through
native `/v1/systemone` on the publisher's pinned Decision2 llama.cpp runtime
commit `c9f5974870fc7982ab2f424bc0840af256189719`. It uses the Decision2
head, without token generation or language-model-logit approximations. The
GGUF contains no decision temperature keys, so the native runtime uses its
T=1.0 fallback. `lux-gguf-contract.json` records this inspection. The native
server formats each question independently and rejects context overflow.
[The server log](lux-q4-quick8.server.log) preserves its model metadata and
runtime diagnostics.

Native option serialization differs: K2 preserves Python insertion order
`yes`, `no`; Lux's native nlohmann JSON parser iterates criteria in lexical
order `no`, `yes`. Question and option meanings are identical. We retain the
publisher interfaces and do not sweep option orders.

Both map positive/negative binary probabilities to the four-state distribution
using the existing independence product. The engine selects hard assessments
and computes exact query probabilities from those assessments and the program's
probabilistic choices. Query MAE measures these resulting query probabilities;
it is not a calibration score for the models' soft evidence distributions.

## Execution and checks

Downloads, isolated dependency setup, and the Lux runtime build overlapped.
Measured model runs are sequential, each with four CPU threads, after CPU-heavy
setup completed. There is no GPU and the memory limit is 16 GiB. Wall times
include model startup and file verification, but exclude downloads/builds.
The archived H2O time is historical: the same thread count does not establish
identical host performance or permit a controlled speedup claim.

Synthetic smoke tests preserve raw distributions, repeat results, group versus
single-question differences, and resource use. K2 repeated outputs are bitwise
identical; group versus single distributions differ by at most 0.004779 with
unchanged hard labels. Lux repeats and group/single results are identical in
this smoke. These checks establish interface behavior on a tiny synthetic
world, not general determinism or accuracy.
The K2 CPU path is not numerically cross-validated against the publisher's CUDA
server; its pointer-head computation runs in float32 without CUDA autocast.

Complete prediction, trace, metadata, resource, and metric files are preserved
for each completed model. Traces include the full four-state distributions and
replayed candidate request groups. Metadata records pinned model revisions,
source/weight hashes, runtime identities, and executed driver hashes.

## Reproduction

Create an isolated Python environment, install `requirements.lock.txt` with
`--extra-index-url https://download.pytorch.org/whl/cpu`, then install
this repository editable. Download the pinned snapshots listed in
`protocol.json`. Build the Lux runtime using `environment.json`'s CMake flags.
Keep the `PYTHONPATH` ordering below: the driver reuses `RecordedBackend` and
`read_rows` from `experiments/question_isolation/run_dev.py`, which shares its
filename with another experiment's driver.

```bash
export PYTHONPATH=experiments/question_isolation:experiments/prompt_v2:experiments/additional_models
python experiments/additional_models/smoke.py \
  --model k2 --model-path /path/to/pinned-k2 \
  --output /path/to/k2-smoke.json
python experiments/additional_models/run_quick.py \
  --model k2 --model-path /path/to/pinned-k2 \
  --output /path/to/new-k2-run --threads 4
python experiments/additional_models/smoke.py \
  --model lux-q4 --model-path /path/to/Decision-2.0-Lux-9B-Q4_K_M.gguf \
  --server-binary /path/to/pinned-llama-server \
  --output /path/to/lux-smoke.json
python experiments/additional_models/run_quick.py \
  --model lux-q4 --model-path /path/to/Decision-2.0-Lux-9B-Q4_K_M.gguf \
  --server-binary /path/to/pinned-llama-server \
  --output /path/to/new-lux-run --threads 4
```

The executed driver retains the original quick8 driver's historical docstring;
this report and `protocol.json` describe the current request. The first K2
benchmark launch had an import-path error before any predictions or run output
were created; correcting the command's `PYTHONPATH` resolved it without changing
the executed driver. Transformers emits documentation warnings about the
publisher's `cache_position` argument; loading and inference complete normally.

Both model cards declare Apache-2.0. Primary sources:
[IFM K2](https://huggingface.co/IFM/K2-Type-0.9B),
[Lux GGUF](https://huggingface.co/vllm-sr/Decision-2.0-Lux-9B-GGUF), and
[Decision2 runtime](https://github.com/Xunzhuo/llama.cpp/tree/c9f5974870fc7982ab2f424bc0840af256189719).
