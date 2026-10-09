# GLiNER attention probe

`probe.py` loads `fastino/gliner2.5-small-v1` at the pinned model revision and
measures interactions between classification schema slots. `results.json`
contains the initial observed results, including model and library revisions.

The experiment compares a task alone, tasks jointly encoded in one schema,
reordered tasks, changed later labels, and independent examples in a batch.
It also records per-layer masks and attention weights. A controlled change
swaps two later label tokens without shifting any positions.

## Observed result

All 12 DeBERTa layers allowed full bidirectional attention among non-padding
tokens. The 81-token example used an all-ones 81-by-81 mask. Earlier and later
schema slots attended to one another. Changing only two later tokens changed
the first task's logits by up to 0.00736. Repeating the identical input yielded
identical logits in the tested runtime.

One task's positive-label logit was 4.876 alone, 4.876 in a separate batch row,
and 3.269 when jointly encoded with three other tasks. This demonstrates
interaction, not an accuracy improvement or a calibrated joint distribution.

## Reproduce

Install in a dedicated environment; the CPU wheel is approximately 187 MiB and
the model weights approximately 296 MB. No API credentials are required.

```bash
python -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu \
  -r experiments/gliner_attention/requirements.lock.txt
.venv/bin/python experiments/gliner_attention/probe.py \
  --output runs/gliner_attention/results.json
```

The pinned GLiNER source includes a tokenizer metadata compatibility fallback;
`protobuf` and `sentencepiece` are included in the lock file. Model weights
are downloaded to the normal Hugging Face cache and are not committed.

The probe explicitly compiles each schema to avoid an order-insensitive
classification facade cache hiding the task-order perturbation. CPU timings
in the result are individual calls, not a performance benchmark.
