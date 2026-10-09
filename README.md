# Relational Decisions

Experiments toward a relational programming engine with natural-language
predicates, local decision models, and probabilistic inference.

The first runtime implements miniKanren-style interleaved search with suspended
neural goals. A deterministic scheduler groups ready predicates for model
evaluation. Memoized goal tables handle recursion, and canonical decision
diagrams preserve the identity of uncertain facts across overlapping proofs.

The supported language is finite positive Datalog with categorical choices,
explicit negative evidence, and registered ground language predicates. This is
not a complete Prolog or miniKanren implementation. See the
[runtime design](docs/engine.md) for semantics and limits.

## Current contents

- [`src/relational_decisions`](src/relational_decisions): the engine, model
  adapters, exact-input decision cache, and benchmark command.
- [`tests`](tests): arithmetic, recursion, scheduling, cache, and differential
  checks against all open benchmark programs and additional generated cases.
- [`benchmarks/nl_logic_v1`](benchmarks/nl_logic_v1/README.txt): 240 synthetic
  programs across 12 reasoning families and three everyday domains, an exact
  reference evaluator, and a scorer.
- [`experiments/gliner_attention`](experiments/gliner_attention/README.md): a
  reproducible local GLiNER2.5 Small attention probe and its observed results.

## Benchmark

| Partition | Programs | Underlying worlds |
| --- | ---: | ---: |
| Train | 144 | 72 |
| Dev | 48 | 24 |
| Held out | 48 | 24 |

Each world has two equivalent text renderings, kept in the same partition.
Entity names and wording vary across partitions. All partitions use the same
12 program families; heldout tests new instances within those families.

Primitive assessments distinguish **supported**, **refuted**, **both**, and
**unknown**. Exact query probabilities follow from explicitly specified
categorical choices and finite positive-Datalog rules. They are not classifier
confidence scores. See the [task contract](benchmarks/nl_logic_v1/task.json).

Development uses train and dev. The heldout archive remains local and ignored
by Git; its frozen SHA-256 is recorded in the
[manifest](benchmarks/nl_logic_v1/manifest.json). A fresh clone intentionally
does not contain heldout prompts or answers. Preserve that archive separately
and freeze the evaluation protocol before a final heldout run.

## Validate and score

Python 3.12 is the tested version. Generation, scoring, and the reference engine
use only the standard library. ProbLog is optional for independent validation.

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
python benchmarks/nl_logic_v1/validate.py
python -m pip install -r benchmarks/nl_logic_v1/requirements.txt
python benchmarks/nl_logic_v1/validate.py --problog
```

Validation checks open data and available artifact checksums. It reports a
missing local heldout archive without opening or regenerating it. It fails if
any required open artifact is absent or any available checksum differs.
The initial 96 train/dev worlds agree with ProbLog within floating-point error.

```bash
python benchmarks/nl_logic_v1/score.py --split dev --predictions predictions.jsonl
```

Use `--derive-probabilities` to isolate language grounding: the reference engine
then calculates downstream probabilities from the submitted assessments.

The [example](benchmarks/nl_logic_v1/example.dev.json) and
[ProbLog export](benchmarks/nl_logic_v1/example.dev.pl) are from dev.

## Run the engine

Start with a [small example](examples/frontier.py) that exposes two ready
predicates in one batch and a third predicate in the next:

```bash
python examples/frontier.py
# probability: 23/25; batches: [move, repair], then [vacancy]
```

An oracle backend exercises the entire scheduler using known language labels:

```bash
rd-bench --backend oracle --split dev --output runs/oracle-dev
```

For local model decisions, install the pinned CPU dependencies documented in
the [attention experiment](experiments/gliner_attention/README.md), then run:

```bash
rd-bench --backend gliner --split dev --schedule frontier \
  --batch-size 4 --search-quantum 64 --output runs/gliner-frontier-dev
```

`--schedule single` scores one proposition per call. `--schedule full` scores
the entire registered schema when the first neural goal is requested.
`frontier` batches currently ready goals, up to the declared batch size.

`--prompt` selects a versioned grounding formulation. The default is the original
`explicit-reports-v1`; experiments also support `direct-status-v2`,
`concrete-options-v2`, and `binary-reports-v2`. The binary formulation asks two
yes/no questions per proposition, for its positive and negative reports. Its
four category weights are a product of the two answers, introducing an extra
independence assumption in soft mode. Prompt definitions and label mappings
are part of the cache identity.

The [prompt experiment](experiments/prompt_v2/README.md) compares formulations
on train and evaluates the selected one with Decide 340M and Decide 1B on dev.
The 1B checkpoint needs the newer dependencies pinned in that experiment;
the adapter rejects older runtimes that would misread its rotary configuration.

The default `--mode hard` uses the selected evidence category as an observed
fact. `--mode soft` gives each proposition a four-outcome random variable using
the normalized model scores. Soft mode assumes independent choices across
propositions; the scores are uncalibrated and are not a learned joint model.

Each run writes predictions, exact-input cache keys, batch traces, explanation
diagrams, metrics, and model/runtime metadata under its output directory. Run
directories must be new. Development commands accept only train and dev.

For complete benchmark metrics the runner also assesses any candidates not
needed by the queries; those calls are marked `complete_assessments` in traces.
The library's `Engine.run()` is lazy by default.

## Findings so far

GLiNER2.5 Small jointly encodes schema slots and text using bidirectional
attention. Changing later slots changes earlier slot representations and
scores. Separate examples in a batch do not share this attention. Cache keys
must therefore include the complete ordered schema and world input when
jointly scoring predicates.

The first probe used a 73.9M-parameter model on CPU in FP32. The development
box has a 16 GiB RAM limit, four CPU cores of quota, and no GPU. GLiNER2.5-Decide
(marketed as 340M) also runs locally in FP32: its dev evaluation took 7.0 minutes
after initialization and peaked at 4.24 GiB process RSS.

The first engine matches all 192 open programs exactly with oracle assessments
in both hard and soft modes. The first GLiNER Small development baseline reaches
22.05% four-way assessment accuracy and 0.2007 query-probability MAE. Under the
same settings, Decide reaches 23.61% and 0.1985 respectively, and never predicts
"supported". Both remain below the 32.64% majority-label accuracy baseline, so
the current formulation is not yet a reliable language grounding system. The
[baseline report](experiments/engine_v1/README.md) preserves predictions,
traces, and the full comparison. Heldout remains unused.

Licensed under the [Apache License 2.0](LICENSE).
