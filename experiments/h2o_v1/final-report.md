# H2O-Lightning final heldout evaluation

The user selected H2O-Lightning-4B after the eight-program development
diagnostic and authorized a final heldout run on 2026-10-09. The system is
frozen in [final-protocol.json](final-protocol.json) before opening any archive
members. Status: protocol frozen; final evaluation pending.

A preflight attempt stopped before reading inputs because the runner compared
an in-memory tuple with its JSON list representation. The comparison was fixed
and the runner hash re-frozen before opening heldout; the selected model,
prompt, precision, engine configuration, and metrics did not change. The failed
preflight produced zero heldout predictions and did not read gold.

The final run covers all 48 heldout programs (24 underlying worlds, two
paraphrases each), 576 proposition assessments, and 272 queries. It uses the
same pinned checkpoint, native choice prompts, temperature 0.8, BF16 backbone,
FP32 answer readout, and hard-mode engine as the diagnostic. There is no
additional prompt selection, calibration, threshold tuning, or model selection.

The live frontier uses the existing defaults: four candidates per request and
search quantum 64, with complete assessments enabled. Each H2O binary question
is evaluated independently. Unlike the dev comparison, heldout has no earlier
GLiNER request trace to replay. Task names and sibling questions do not appear
in this backend's native prompt, so live request grouping does not change the
model question. No shared-prefix optimization is introduced.

The archive must match its original manifest checksum. The runner verifies
the frozen source hashes and runtime identity before reading inputs. It saves
each completed program atomically, then seals all submitted predictions with
a SHA-256 checksum before reading the gold member. It reports accuracy,
macro-F1, query MAE, exact-query rate, family/domain breakdowns, and paraphrase
agreement. Every submitted query is also checked against the independent exact
evaluator given predicted facts; gold is checked against its oracle facts.

`final_eval.py` is a dedicated final-evaluation entry point. The ordinary
development CLI continues to accept only train/dev. A test using public dev
fixtures checks the checkpoint/recovery path and verifies that gold cannot be
read before all predictions are saved. Recovery is permitted only for technical
interruptions with the identical frozen configuration, never for improving a
heldout score.

```bash
python experiments/h2o_v1/final_eval.py --output runs/h2o-final-heldout-v1
```

Run in the same pinned environment as the dev diagnostic. Add `--resume` only
for a technical interruption of that output directory. Completed evaluations
cannot be resumed. The raw inputs, gold, predictions, cache, checkpoints, and
detailed traces remain local; the aggregate final report and provenance are
published. Once scored, this partition is an evaluated final test, no longer
an untouched heldout for selecting future improvements.
