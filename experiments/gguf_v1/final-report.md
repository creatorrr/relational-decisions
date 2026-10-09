# H2O Q8 final heldout result

The complete frozen evaluation finished on 2026-10-09. H2O-Lightning Q8_0
achieves **87.5% assessment accuracy** and **250/272 exact query probabilities**
on 48 heldout programs from 24 underlying worlds. This is close to the 88.54%
grounding result on the eight-program dev diagnostic. Contradictory reports
remain the main weakness.

| Measure | Heldout result |
| --- | ---: |
| Correct proposition assessments | 504/576 (87.50%) |
| Assessment macro-F1 | 0.85446 |
| Exact query probabilities | 250/272 (91.91%) |
| Query probability MAE | 0.03461 |
| Largest query error | 1.0 |
| Assessment agreement between paraphrases | 87.85% |
| Mean query difference between paraphrases | 0.04746 |
| Complete run wall time | 59.09 min |
| Peak server RSS | 4.65 GiB |
| Peak Python client RSS | 0.50 GiB |

The memory peaks are measured separately; their sum is a conservative bound,
not a measured simultaneous peak. Runtime covers startup, all inference,
checkpointing, scoring, and exact-reference verification. Earlier stopped
attempts are excluded from this run's runtime and score and are documented
separately.

## What the errors mean

Of 132 contradictory assessments, 62 are classified as both, 32 as supported,
37 as refuted, and one as unknown. These account for **70 of 72 assessment
errors**. The other two errors classify supported reports as unknown.
Supported-only, refuted-only, and unknown cases collectively score 442/444
(99.55%).

The `contradictory_reports` family has the largest query MAE, 0.18333. Binding
joins, cyclic reachability, left recursion, and missing-versus-negative queries
all have zero query error. Query accuracy can hide primitive mistakes when the
affected facts do not change a particular query. Both proposition and query
metrics matter. Equivalent paraphrases still cause some different judgments,
so contradiction handling and wording robustness are useful next targets.

## Frozen evaluation and verification

The selected model, binary reports prompt, T=0.8 A/B readout, prefix-reuse
configuration, hard inference mode, four-candidate frontier, and search quantum
64 are frozen in [final-protocol-v2.json](final-protocol-v2.json). The protocol
was committed at `a508949` before this run opened its inputs. No settings
were changed during the complete run.

All 48 predictions were saved and sealed at `2026-10-09T08:46:12.965445Z`.
Gold was opened afterward, at `2026-10-09T08:46:12.967061Z`. The prediction
checksum is
`26673ffc5f0093a1778e5b5d4c0348067a488debcf1c2440442faa6dd97b78c2`.

The independent exact evaluator verified all 272 submitted query probabilities
given predicted facts, and verified all 48 gold programs against their oracle
facts. A separate Fraction-equality check confirms that the 250 matching query
probabilities are exactly equal as rationals; the count does not depend on the
scorer's floating-point tolerance. The remaining errors arise from language
grounding in this benchmark.

The original BF16 attempt was stopped at the user's request before scoring.
The first Q8 attempt stopped on a completion-transport bug after four programs,
also before gold was read. That bug was reproduced and corrected on public dev
without changing raw model scores. This complete v2 run started all 48 programs
again with fresh caches. See [the experiment notes](README.md).

Aggregate results and provenance are in `final.metrics.json`,
`final.metadata.json`, `final.resources.json`, `final.seal.json`, and
`final-exact-check.json`. Heldout inputs, gold, predictions, detailed traces,
and caches remain local and ignored by Git. This partition has now been
evaluated; future model selection should use dev and a fresh final test.

```bash
export LLAMA_SERVER_BIN=/path/to/llama-b11515/llama-server
python experiments/h2o_v1/final_eval.py \
  --protocol experiments/gguf_v1/final-protocol-v2.json \
  --output runs/h2o-q8-final-heldout-v2
```

This command requires the original local archive, the pinned Python environment,
and matching model/runtime checksums. It refuses to overwrite a completed run.
