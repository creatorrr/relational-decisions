# Does isolating questions improve GLiNER?

This control was requested after the first 16 OpenDecision dev programs showed
better proposition accuracy. That comparison changed the checkpoint, whether
questions shared attention, and where questions appeared in the input. These
controls separate the last two changes on the existing GLiNER 340M checkpoint.

The initial protocol froze two additional dev runs before their predictions
were observed. At the user's request we then stopped the exhaustive sweep:
the native-isolation run completed 24 programs, and the remaining comparisons
use only the first eight dev programs as a quick diagnostic. Neither is used
to select a prompt. Heldout remained sealed during these controls. The subsequent
[H2O Q8 final evaluation](../gguf_v1/final-report.md) is complete.

## Practical notes

Isolating questions alone did not help. On the 24 completed programs (288
assessments), the shared-question baseline scores **27.08%**, while isolated
native questions score **23.96%**. Macro-F1 falls from 0.1990 to 0.1089; query
MAE rises from 0.1817 to 0.1881. The isolated model predicts `unknown` 284 times
and `supported` four times. All 133 resulting query probabilities match the
independent reference given those predicted facts.

These partial results are preserved under `gliner-isolated-native-partial.*`
and compared on exactly the same input IDs in `partial-comparison.json`.
The run was interrupted during the next program, so its resource measurement
includes unfinished work. It is not a complete 48-program evaluation.

The eight-program follow-ups retain the frozen wording and original request
groups. They check question-in-text GLiNER and the independent OpenJev model.
This deliberately shortened diagnostic is not a representative or heldout
estimate, and the first eight records do not form complete paraphrase pairs.

The matched quick results (96 assessments and 42 queries) are:

| Configuration | Assessment accuracy | Macro-F1 | Query probability MAE |
| --- | ---: | ---: | ---: |
| GLiNER 340M, shared questions | 30.21% | 0.2119 | 0.1197 |
| GLiNER 340M, isolated native questions | 19.79% | 0.0826 | 0.1387 |
| GLiNER 340M, isolated question in text | 22.92% | 0.1245 | 0.1387 |
| GLiNER 1B, shared questions | 26.04% | 0.2533 | 0.0972 |
| OpenDecision, isolated question in text | 43.75% | 0.3846 | 0.1195 |
| OpenJev, isolated native questions | 39.58% | 0.3395 | 0.1170 |
| H2O-Lightning-4B, isolated native questions | 88.54% | 0.8403 | 0.0147 |

The majority-label accuracy is 39.58% on this small, unbalanced prefix.
OpenJev ties that accuracy, while its macro-F1 exceeds GLiNER's. OpenDecision
gets four more assessments right than OpenJev. These are useful diagnostic
signals, not enough to rank generalization. Every query output in this table
matches the independent reference given that model's predicted facts.

The later [H2O-Lightning diagnostic](../h2o_v1/README.md) uses the same eight
inputs and frozen question wording, with its native causal prompt and CPU
BF16 backbone/FP32 scores. It gets 85/96 assessments and 40/42 ground-truth
query probabilities right. All eleven assessment errors collapse contradictory
evidence to one side. This is an encouraging practical improvement; the larger
model and different runtime do not make it an architecture-only comparison.

**Working conclusion:** neither isolation alone nor transferring the question
into the text reproduces OpenDecision's advantage on GLiNER in these checks.
OpenDecision remains the completed full-dev baseline; H2O is the strongest
candidate in this small diagnostic. There is no evidence here to justify changing the scheduler to
serialized paths. Relevant path context remains a separate, untested idea.

`quick-comparison.json` contains the exact input IDs and metrics;
`summarize_quick.py` regenerates it from the archived predictions. The original
`summarize.py` is for a future completed full sweep and is not the audit for
this deliberately shortened experiment. Full-run protocols are retained as
the historical plan; this section records the actual scope.

The short state-layout run took 193.03 seconds including loading, with 4.13 GiB
peak RSS. OpenJev took 186.26 seconds with 2.46 GiB peak RSS. Both were fresh
CPU FP32 runs. The probes verified bitwise repeated-call determinism; batched
versus serial results matched exactly for the tested GLiNER inputs and within
2.24e-7 in category weights for OpenJev, with identical hard labels.

| Condition | Questions in one attention sequence | Question placement |
| --- | --- | --- |
| Shared baseline | Up to eight | Original GLiNER schema |
| Isolated native | One | Original GLiNER schema |
| Isolated question in state | One | Before world text, matching the OpenDecision adapter |

Both isolated conditions can process eight independent rows in a single CPU
batch. They do not share attention across rows. Within a binary question,
GLiNER's yes/no labels still share attention; OpenDecision encodes answer
options separately. The final condition also changes the task name to `answer`,
matching OpenDecision, so it tests that complete layout change rather than
distinguishing name effects from question placement.

## Controlling search-dependent grouping

The driver replays the exact ordered candidate groups from the archived GLiNER
340M baseline. This preserves original slot names and avoids changed predictions
altering later request groups. It obtains all model assessments first, then runs
the unchanged hard-mode engine against those recorded distributions. Model
request traces and later symbolic lookup traces are recorded separately.

This is an offline intervention on the original model requests. It does not
measure the latency of a new live frontier scheduling policy. The core engine,
world text, binary wording, answer descriptions, checkpoint, precision,
temperature, package versions, and four-category mapping stay fixed.

`probe.py` checks that the original joint scorer reproduces one archived baseline
group exactly, and that independent-row batching agrees with scoring the same
questions in separate calls within numerical tolerance. The full audit checks
every query against the independent exhaustive reference.

## Reproduce

Install the pinned environment from
[`../opendecision_v1/requirements.lock.txt`](../opendecision_v1/requirements.lock.txt),
then run from the repository root:

```bash
python experiments/question_isolation/probe.py --output runs/isolation-probe.json
python experiments/question_isolation/run_dev.py --layout isolated-native \
  --output runs/gliner-isolated-native-dev
python experiments/question_isolation/run_dev.py --layout isolated-question-in-state \
  --output runs/gliner-isolated-state-dev
```

Output and cache directories must be new. Each run saves predictions, traces,
metrics, immutable model/source identities, timing, and peak process RSS.

Those commands reproduce the originally planned full sweeps. The actual quick
follow-ups use:

```bash
python experiments/question_isolation/run_quick.py --model gliner-state \
  --output runs/gliner-state-quick8
python experiments/question_isolation/run_quick.py --model openjev \
  --output runs/openjev-quick8
```

If question isolation helps, a later experiment can test a bounded path through
the symbolic rules as relevant context for a single target proposition. That
hypothesis is separate from isolation: this experiment does not evaluate paths
or change how shared facts and overlapping proofs receive probabilities.
