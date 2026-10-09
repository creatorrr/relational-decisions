# Local GGUF decisions

The [final heldout evaluation](final-report.md) is complete: 504/576 correct
assessments (87.50%), 250/272 exact queries (91.91%), and query MAE 0.03461.

The user requested H2O-Lightning and LiquidAI d1 GGUF trials before continuing
the final evaluation. The previous BF16 heldout attempt was stopped without
reading gold or scoring partial predictions. Both GGUFs are tested on the same
first eight public dev programs used in the earlier comparison. No prompts,
thresholds, or label definitions are changed.

We start with Q8_0 for both models. IQ4_XS/Q6_K for H2O and Q4_K_M for d1 are
available but have not been evaluated. GGUF is a format; smaller weights do
not by themselves guarantee faster prompt processing on this CPU.

## Runtime and scoring

- llama.cpp **b11515**, commit `3d65c90d04d337e88f2b1f7f0061f40a5324e662`,
  Ubuntu x64 CPU release. b11429 cannot load the d1 decision-model type.
- Four CPU threads, no GPU; 16 GiB memory limit. Experiments run sequentially.
- H2O Q8_0: third-party `mradermacher/h2o-lightning-4b-GGUF`, revision
  `7a62fde885ed87e7b22a2195099c6b204381138f`. Original conversion's precise
  upstream weight revision is not established; this is not a clean quantization
  ablation. Published H2O rendering is pinned separately to
  `h2oai/h2o-lightning-4b@672dc01ed37a516357cd3c7da777c96699d3f16c`.
- H2O reads verified ` A`/` B` token log probabilities from `/completion`,
  before sampling filters, and applies the published label softmax at T=0.8.
  The generated token is unused. Tokenization must match the pinned HF tokenizer;
  missing label scores or truncation cause failure.
- Each H2O group starts by resetting and prefilling its exact common prefix.
  Questions then reuse that prefix. The hybrid recurrent model needs short
  state checkpoints (`--checkpoint-min-step 0`). Prefix reuse introduces small
  numerical differences, so it is part of the frozen runtime identity.
- d1 Q8_0: official `LiquidAI/d1-3B-GGUF`, revision
  `bb1e436ea78eb96a3f1acb6da865f70c2fbeb563`, through `/v1/systemone`.
  Eight slots let up to eight binary questions share a prefix and branch into
  independent causal suffixes. There is no bidirectional question interaction.
  Native readout takes the maximum over single-token aliases of each code,
  then softmax. The GGUF contains no temperature metadata: the server uses T=1.
  Both models use the same frozen positive/negative questions and product mapping
  into supported/refuted/both/unknown.

Model checksums, server binary/library checksums, exact flags, prompt identities,
and readout settings are captured in each run's metadata. Context overflow is
rejected. Server peak RSS is reported separately from Python client peak RSS;
the sum of peaks is a conservative bound, not a simultaneous peak measurement.

## Initial smoke checks

`smoke.json` records the same Oak/Elm toy world for both models. H2O's warm-prefix
repeats and d1's grouped repeats were bitwise identical. d1 individual versus
grouped probabilities differed by at most about 0.00048; all four hard decisions
agreed. H2O cold versus prefix-reused probabilities also differed slightly with
the same four hard decisions. This is a small empirical check, not a universal
cross-hardware determinism guarantee.

The GGUF SystemOne template and llama.cpp d1 rendering/readout were checked
against the publisher's `prompt.py`. d1 is causal LFM2, combining causal
convolution and attention. H2O is causal Qwen3.5 with Gated DeltaNet and attention.

## Run the matched dev check

Use the Python environment pinned in `../h2o_v1/requirements.lock.txt`, install
this repository editable, and extract the llama.cpp b11515 Ubuntu x64 release.
Only H2O's tokenizer/shim requires Transformers; d1 uses the native HTTP endpoint.
The adapter owns a local server, verifies model bytes, and stops it on exit.

```bash
export LLAMA_SERVER_BIN=/path/to/llama-b11515/llama-server
export PYTHONPATH=experiments/prompt_v2:experiments/question_isolation
python experiments/question_isolation/run_quick.py --model d1-q8 --output runs/d1-q8-quick8
python experiments/question_isolation/run_quick.py --model h2o-q8 --output runs/h2o-q8-quick8
```

## Matched dev results and selection

| Runtime | Correct assessments | Macro-F1 | Exact queries | Query MAE | Wall time | RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2O BF16, independent forwards | 85/96 | 0.8403 | 40/42 | 0.01467 | 17.88 min | 7.39 GiB |
| H2O Q8_0, prefix reuse | 85/96 | 0.8403 | 40/42 | 0.01467 | 8.89 min | 4.61 GiB server + 0.49 GiB client |
| d1 Q8_0, native branches | 73/96 | 0.7234 | 41/42 | 0.00514 | 4.28 min | 3.08 GiB server + 0.03 GiB client |

H2O Q8's hard assessments and exact query outputs match BF16 on every one of
these eight programs. The runtime is 2.01 times faster including startup and
uses substantially less memory. This combines quantization, a different
inference implementation, and prefix reuse; the speed gain is not attributed
to quantization alone.

d1 is faster and has one more exact query on this subset, but makes 23 grounding
errors instead of 11. Downstream queries do not expose every primitive error.
We retain the already selected H2O model, now in Q8_0 with prefix reuse, for
the final heldout run. No further prompt or quantization sweep is performed.
All 42 query outputs for each runtime agree with the independent exact
evaluator given that runtime's predicted facts.

Checked-in `*-quick8.*` files contain dev results only. The d1 run used the
driver before formatting at commit `01dce02`; H2O used `19407ad`. Formatting
changed the driver checksum, with no behavioral change.

The replacement final protocol and final report are saved alongside this
experiment. Gold is read only after all 48 programs' predictions are sealed.

### Completion transport correction before scoring

The first Q8 heldout attempt stopped after four completed programs because a
completion response omitted its token-probability record. Gold remained unread;
partial predictions were not scored or used for selection. The interruption
is recorded in `final-v1-interruption.json` and its original protocol is retained.

llama.cpp omits a token record when the unused sampled token leaves an incomplete
UTF-8 character. The original empty sampler list also failed to explicitly
select greedy sampling. The corrected transport constrains the unused token to
ASCII with `root ::= " A"` and enables the temperature sampler. Scores still
come from the server's raw model logits before grammar, bias, or sampling.
Neither the question prompt nor the A/B readout changes.

`verify_score_control.py` reproduces the missing record on a public dev prompt
by forcing an incomplete UTF-8 byte token. ASCII grammar restores the record;
the raw A/B probabilities are bitwise identical even under that artificial
sampler bias. The corrected adapter also preserves all 12 assessments of the
first dev program and its probability distributions within floating-point
roundoff. Results are in `score-control-check.json`. The final v2 protocol
freezes this correction and restarts all 48 programs with fresh caches.
