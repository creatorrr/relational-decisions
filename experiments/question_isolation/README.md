# Does isolating questions improve GLiNER?

This control was requested after the first 16 OpenDecision dev programs showed
better proposition accuracy. That comparison changed the checkpoint, whether
questions shared attention, and where questions appeared in the input. These
controls separate the last two changes on the existing GLiNER 340M checkpoint.

The protocol freezes two additional dev runs before their predictions are
observed. Neither is used to select a prompt. Heldout remains sealed.

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

If question isolation helps, a later experiment can test a bounded path through
the symbolic rules as relevant context for a single target proposition. That
hypothesis is separate from isolation: this experiment does not evaluate paths
or change how shared facts and overlapping proofs receive probabilities.
