NL Logic Bench v1
=================

Purpose
-------
A small synthetic test bed for a relational engine with natural-language
primitive predicates and probabilistic choices. This is a correctness and
composition benchmark, not a broad natural-language-understanding benchmark.
No model has been evaluated on this dataset at creation.

Contents and split
------------------
120 underlying worlds, each with two equivalent text renderings: 240 programs.
Train: 72 worlds / 144 programs. Dev: 24 worlds / 48 programs.
Heldout: 24 worlds / 48 programs, stored locally in sealed/heldout-v1.tar.gz.
The sealed directory is ignored by Git; preserve the original archive apart
from the repository. Fresh clones intentionally lack this archive.
Each world has 12 candidate propositions and several ground queries.
Rooms, deliveries, and gardens provide the three everyday domains.

The split is by underlying world, not by sentence, query, or text rendering.
Paired renderings always stay in one partition. Probabilities, signed facts,
and entity bindings are identical within each pair. Entity names and text
realizations use separate vocabularies/templates for train, dev, and heldout.
Generation rejects repeated semantic configurations modulo entity renaming.
All 12 reasoning families occur in each split: this tests new instances and
surface forms within known families, not completely unseen program grammars.
The 48 heldout programs are 24 independent world groups, not 48 independent
problems; statistical resampling should use groups.

The heldout partition was assigned before model evaluation. The manifest
freezes SHA-256 hashes, including the heldout archive. Development tools read
only train and dev; validation hashes the sealed archive without opening it.
Use train for fitting/prompts and dev for choices and error analysis. Freeze
the engine, prompts, calibration, and batching policy before final heldout
evaluation. Do not choose improvements based on heldout outcomes. If a new
benchmark version is needed, preserve this archive and label the new version.
The archive is organizational separation, not cryptographic access control.

Ground-truth semantics
----------------------
1. world_text is a collection of unordered explicit reports about one snapshot.
   No report is more authoritative or newer than another. These are reports,
   not guaranteed physical truths. The input candidate proposition and its
   explicit negative describe what to look for in those reports.

2. Classify each candidate as:
      supported: its positive assertion is reported, its negative is not;
      refuted:   its negative assertion is reported, its positive is not;
      both:      both assertions are reported;
      unknown:   neither assertion is reported.
   Add candidate.atom for supported or both. Add candidate.negative_atom for
   refuted or both. Unknown adds neither. The negative atom is an ordinary
   explicit predicate, e.g. not_ready(X); it is not negation-as-failure.
   Contradictions do not entail unrelated atoms. Missing text is not false.

3. Merge these signed observations with program.facts, which are trusted
   structured facts. Apply finite positive, function-free Horn rules to their
   least fixed point. Variables start with '?'; all other terms are constants.
   Every rule's head variables occur in its body. Recursion is supported.
   Queries are ground; existential joins are represented in rule bodies.

4. Each program.choices element is one categorical random variable. Exactly
   one outcome is selected, with the listed rational probability. Distinct
   choices are independent. An outcome can add zero or more facts. Reusing a
   choice's fact in multiple proofs never resamples it. A Bernoulli choice is
   represented by a fact-producing outcome and an empty alternative.

5. program.evidence lists Boolean conditions on logical atoms. Query answers
   are conditioned on these, after rule closure. This is distinct from the
   natural-language evidence classification in step 2. Each generated case
   has nonzero conditioning probability.

6. Gold query probabilities are exact fractions from enumerating all choices,
   computing the fixed point in each possible world, retaining worlds that
   satisfy evidence, and normalizing their weights. They are not fabricated
   classifier confidence scores. Most programs have 8 possible worlds; the
   categorical family has 24. This keeps the exact reference cheap.

Families
--------
conjunction          independent prerequisites
disjunction          alternative proofs, including overlapping success worlds
shared_diamond       two proofs sharing one probabilistic prerequisite
repeated_choice      using the same uncertain fact twice must not square p
exclusive_choice     categorical outcomes cannot coexist
staged_chain         multiple stages revealing additional primitive goals
cyclic_reachability   transitive closure over a cyclic probabilistic graph
left_recursion       recursive clause before its base case; finite fixed point
binding_join         shared-variable joins and argument direction
contradictory_reports explicit positive and negative reports without explosion
missing_vs_negative  absence and explicit opposition produce different results
conditioning         Bayes conditioning on an event with overlapping causes

Files and use
-------------
data/{train,dev}.inputs.jsonl contains only model/engine inputs.
data/{train,dev}.gold.jsonl contains assessments, oracle facts, and probabilities.
Never concatenate the gold records into model input. Candidate IDs and program
IDs are identifiers; they are not labels. The header semantics above should be
fixed across model variants when constructing their task descriptions.

example.dev.json is one open example with its gold for inspection.
example.dev.pl exports the same example, including oracle facts, as ProbLog.
manifest.json contains counts and frozen output hashes.
validation.json records the initial open-split checks.
logic.py is the dependency-free exact reference evaluator.
generate.py refuses to overwrite already generated data.

Run structural, arithmetic, and independent ProbLog validation from this folder:
  python validate.py --problog
Missing local heldout archives are reported; available artifacts must match
their manifest hashes. Validation never opens or regenerates heldout.

Evaluate a JSONL prediction file:
  python score.py --split dev --predictions predictions.jsonl

Each prediction record has this shape:
  {"id": "...", "assessments": {"p00": "supported", ...},
   "query_probabilities": {"act_0": "3/10", ...}}

All input IDs, candidate IDs, and query IDs must be covered. Fractions or
decimal probabilities are accepted. Missing or extra predictions are rejected.
To isolate language grounding, supply assessments and derive probabilities
with the fixed reference engine:
  python score.py --split dev --predictions predictions.jsonl --derive-probabilities

Recommended comparisons
-----------------------
Logic-only: supply oracle_facts to a candidate inference engine and compare
its query marginals with gold. This tests search/provenance/probability logic.
Language-only: predict four-way assessments from text and candidate definitions.
End-to-end: feed predicted signed observations through the logic engine.
Scheduling: compare fixed joint schemas, frontier-driven schemas, and separate
batch rows while keeping the rest of the evaluation protocol fixed. Reuse the
same uncertainty identity across branches. Log complete ordered schema inputs.

The scorer reports assessment accuracy, macro-F1, probability MAE, max error,
probabilities correct within 1e-9, family/domain breakdowns, and agreement
between equivalent renderings. Soft model confidences and calibration need a
separate evaluation protocol; they are not ground-truth event probabilities.

Limitations
-----------
Synthetic, small vocabularies, controlled paraphrases, finite domains, and
positive Datalog. No implicit commonsense facts, defaults, actions over time,
function symbols, or unrestricted negation. The language wording is authored
from the semantic specification; natural-language equivalence is not proved
by the arithmetic oracle. Heldout is a modest sanity check, not a precise
estimate of real-world quality.
