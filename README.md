# Relational Decisions

Experiments toward a relational programming engine with natural-language
predicates, local decision models, and probabilistic inference.

The current design direction is miniKanren-style interleaved search with
suspended neural goals. A scheduler groups ready predicates for model
evaluation, while shared symbolic explanations preserve the identity of
uncertain facts across proofs. The runtime itself is not implemented yet.

## Current contents

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

## Findings so far

GLiNER2.5 Small jointly encodes schema slots and text using bidirectional
attention. Changing later slots changes earlier slot representations and
scores. Separate examples in a batch do not share this attention. Cache keys
must therefore include the complete ordered schema and world input when
jointly scoring predicates.

The first probe used a 73.9M-parameter model on CPU in FP32. The development
box had a 16 GiB RAM limit, four CPU cores of quota, and no GPU. Larger-model
capacity estimates have not yet been benchmarked.
