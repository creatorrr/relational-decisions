# First engine development baseline

This is the initial adapter/prompt experiment, recorded without tuning against
the heldout set. It uses all 48 dev renderings from 24 underlying worlds:
576 proposition assessments and 272 queries. Heldout was not opened or run.

The engine was first checked against every train/dev program using oracle
categories, in both hard and soft modes, and against 40 additional generated
programs with nonlinear recursion. All exact query probabilities matched the
independent exhaustive reference. The reference benchmark itself had already
been cross-checked against ProbLog.

## Configuration

- Model: `fastino/gliner2.5-small-v1`, immutable revision recorded in metadata.
- CPU, FP32, four PyTorch threads, deterministic algorithms enabled.
- Prompt: `explicit-reports-v1`, four described labels per proposition.
- Scheduler: frontier, at most four propositions per batch, 64 agenda events
  between opportunities to service pending requests.
- Hard mode: argmax assessment becomes signed observed facts. Probabilistic
  program choices retain their exact declared weights.
- All candidates are eventually assessed for benchmark metrics. Calls solely
  for completing that coverage are marked separately in traces.

## Results

| Metric | Oracle backend | GLiNER Small |
| --- | ---: | ---: |
| Assessment accuracy | 100% | 22.05% |
| Assessment macro-F1 | 1.0000 | 0.0976 |
| Query probability MAE | 0 | 0.2007 |
| Query probabilities within 1e-9 | 100% | 57.72% |

The model predicted `both` for 567 of 576 assessments. A majority-label baseline
that always predicts `supported` would achieve 32.64% accuracy here. The first
adapter/checkpoint therefore fails this proposition-grounding task; high
paraphrase agreement mostly reflects its nearly constant predictions.

The GLiNER run took 55.6 seconds after model initialization, with 156 backend
calls including completion calls. This single CPU run is not a throughput
benchmark. No single-versus-frontier accuracy claim follows from it.

`oracle-dev.metrics.json` and `frontier-dev.metrics.json` include family/domain
breakdowns. Metadata, predictions, and complete frontier traces are preserved.
Traces include normalized scores, exact batch membership, cache keys, query
explanations, and scheduler statistics. The raw model scores are uncalibrated.

## Reproduce

After installing the engine and pinned GLiNER dependencies:

```bash
rd-bench --backend oracle --split dev --output runs/new-oracle-dev
rd-bench --backend gliner --split dev --schedule frontier --mode hard \
  --batch-size 4 --search-quantum 64 --output runs/new-frontier-dev
```

Use a new cache directory with `--cache runs/new-cache` for uncached timing.
The experiment artifacts are development data, not heldout evaluation results.
