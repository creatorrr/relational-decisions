# Engine development baselines

These runs evaluate the initial adapter/prompt without tuning it between
checkpoints. They use all 48 dev renderings from 24 underlying worlds:
576 proposition assessments and 272 queries. Heldout was not opened or run.

The engine was first checked against every train/dev program using oracle
categories, in both hard and soft modes, and against 40 additional generated
programs with nonlinear recursion. All exact query probabilities matched the
independent exhaustive reference. The reference benchmark itself had already
been cross-checked against ProbLog.

## Configuration

- Models: `fastino/gliner2.5-small-v1` and `fastino/GLiNER2.5-Decide` (340M
  according to its model card), immutable revisions recorded in metadata.
- CPU, FP32, four PyTorch threads, deterministic algorithms enabled.
- Prompt: `explicit-reports-v1`, four described labels per proposition.
- Scheduler: frontier, at most four propositions per batch, 64 agenda events
  between opportunities to service pending requests.
- Hard mode: argmax assessment becomes signed observed facts. Probabilistic
  program choices retain their exact declared weights.
- All candidates are eventually assessed for benchmark metrics. Calls solely
  for completing that coverage are marked separately in traces.

## Results

| Metric | Oracle backend | GLiNER Small | GLiNER Decide 340M |
| --- | ---: | ---: | ---: |
| Assessment accuracy | 100% | 22.05% | 23.61% |
| Assessment macro-F1 | 1.0000 | 0.0976 | 0.1707 |
| Query probability MAE | 0 | 0.2007 | 0.1985 |
| Query probabilities within 1e-9 | 100% | 57.72% | 54.78% |
| Paraphrase assessment agreement | 100% | 98.96% | 47.92% |

Small predicted `both` for 567 of 576 assessments. A majority-label baseline
that always predicts `supported` would achieve 32.64% accuracy here. The first
adapter/checkpoint therefore fails this proposition-grounding task; high
paraphrase agreement mostly reflects its nearly constant predictions.

The Small run took 55.6 seconds after model initialization, with 156 backend
calls including completion calls. This single CPU run is not a throughput
benchmark. No single-versus-frontier accuracy claim follows from it.

The `*.metrics.json` files include family/domain breakdowns. Metadata,
predictions, and complete model frontier traces are preserved.
Traces include normalized scores, exact batch membership, cache keys, query
explanations, and scheduler statistics. The raw model scores are uncalibrated.

## Decide 340M comparison

The larger model improves assessment accuracy by only 1.56 percentage points
and remains below the 32.64% majority-label baseline. It predicts `both` 371
times, `unknown` 194 times, `refuted` 11 times, and `supported` zero times.
This checkpoint change does not fix the current four-way grounding formulation.
It does not establish a limit on the model under other formulations or tuning.

Confusion matrix for Decide; rows are ground truth, columns are predictions:

| Truth / prediction | supported | refuted | both | unknown |
| --- | ---: | ---: | ---: | ---: |
| supported | 0 | 1 | 117 | 70 |
| refuted | 0 | 4 | 81 | 45 |
| both | 0 | 5 | 87 | 34 |
| unknown | 0 | 1 | 86 | 45 |

Query MAE improves slightly, but fewer query probabilities match gold within
1e-9 (149/272 versus 157/272). Paraphrase agreement drops substantially; Small's
high agreement was mostly a consequence of its nearly constant prediction.
The two renderings of each world are paired observations, so the 48 programs
should not be treated as 48 independent worlds.

The run took 418.5 seconds after initialization, about 7.5 times Small's elapsed
time. All 159 batches invoked the model with no cache hits. Peak process RSS
was 4,443,024 KiB (4.24 GiB); total wall time including download and initialization
was 441.9 seconds. This was CPU FP32, not INT8. The scheduler policy is identical,
but actual batch membership can change because earlier predictions affect
which goals become ready. These are single-run timings, not controlled
throughput measurements.

Only the checkpoint and its revision changed: inputs, prompt, label ordering,
engine configuration, package versions, and CPU settings match the Small run.
The runtime source was commit `45e30d120b6dad0c66e8cbef803bd29474536c51`.
The independently enumerating reference exactly reproduced all 272 submitted
query probabilities from Decide's predicted assessments. The metrics were
also rescored from the archived predictions. Thus the observed errors here
are in grounding, not a disagreement between the two inference engines.
Heldout was not inspected or evaluated; its archive checksum is unchanged.

Artifacts use the `decide-340m-dev` prefix. The provenance file records the
command and verification; the resources file records Linux process measurements.
The 340M designation comes from the model card. For memory accounting, the
downloaded FP32 checkpoint contains 486,444,053 saved tensor elements across
the encoder and extraction heads; the module totals are in provenance.

## Reproduce

After installing the engine and pinned GLiNER dependencies:

```bash
rd-bench --backend oracle --split dev --output runs/new-oracle-dev
rd-bench --backend gliner --split dev --schedule frontier --mode hard \
  --batch-size 4 --search-quantum 64 --output runs/new-frontier-dev
rd-bench --backend gliner --split dev --schedule frontier --mode hard \
  --model fastino/GLiNER2.5-Decide \
  --revision 9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90 \
  --batch-size 4 --search-quantum 64 \
  --cache runs/new-decide-cache --output runs/new-decide-dev
```

Use a new cache directory with `--cache runs/new-cache` for uncached timing.
The experiment artifacts are development data, not heldout evaluation results.
