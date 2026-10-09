# Engine v0.1

The runtime combines goal-directed relational search with a scheduler that
suspends ground language predicates and scores several ready predicates at
once. Its probability calculation uses symbolic explanations, independently
of the exhaustive possible-world reference in the benchmark.

## Data and language

Programs use the benchmark JSON format: ground facts, positive Horn rules,
independent categorical choices, queries, and optional Boolean evidence.
An atom is an array such as `["ready", "room_a"]`; variables start with `?`.
Terms are flat strings. Rule-head variables must occur in the body, facts and
random-choice outcomes must be ground, and predicate arities must agree.

A `Candidate` registers one ground positive atom, its explicit negative atom,
two natural-language descriptions, and a canonical ID. Equivalent propositions
must share one registration; the engine rejects duplicate atom ownership.
It does not attempt automatic semantic deduplication or extract entities.
An unbound relational goal can enumerate matching registered ground candidates.

Candidate assessments have four mutually exclusive outcomes: supported,
refuted, both, and unknown. Supported/both provide the positive atom;
refuted/both provide its explicit negative. Unknown provides neither.
The model reads the original world text; inferred conclusions are not fed back
as additional observations.

## Search and suspension

Goal tables are keyed by predicate, constants, and variable-equality pattern,
independent of variable names. Rules get fresh variables each time a table is
expanded. Unification extends an immutable-by-convention copied substitution.

The FIFO agenda processes three events:

1. Expand a new goal table with matching facts, choices, candidates, and rules.
2. Resume a rule continuation until its next goal. Subscribe it to that goal's
   table so later answers can wake it.
3. Deliver a new or enlarged answer support to a subscribed continuation.

Repeated states are deduplicated. Answers are ground atoms paired with a
decision-diagram node. When a second proof enlarges an answer's support, the
engine wakes its consumers again. Canonical idempotent union reaches the least
fixed point for finite positive programs, including left and nonlinear recursion.
A cycle with no base case yields no answers.

A neural leaf registers a pending request and a waiter, then yields. After
`search_quantum` agenda events, or earlier if the agenda empties, the scheduler
scores a batch. Its results wake all branches waiting on that proposition.
Other branches are free to expose additional ready predicates before scoring.

Scheduling modes are `frontier`, `single`, and `full`. Frontier requests are
sorted by canonical candidate ID before selecting the next batch. Full mode
scores a fixed complete schema at the first neural request. All policies are
deterministic for fixed program/rule/query ordering, configuration, and scores.
Different scheduling policies are different model experiments: GLiNER's shared
encoder makes the scores depend on which slots appear together and their order.

Each candidate is assessed at most once per run, even if used in many proofs.
The first batch context fixes its assessment. Later ready predicates do not
retroactively rescore earlier ones. This deliberately avoids feedback loops and
changing the probability model halfway through a query.

## Probabilistic explanations

The engine uses a reduced ordered multi-valued decision diagram. Each variable
is one categorical choice, and its outgoing edges correspond to outcomes.
Repeated variable names represent the same choice, not fresh draws.

Conjunction combines explanation nodes with AND; alternate proofs combine them
with OR. Identical nodes are interned and redundant tests eliminated. Mutually
exclusive outcomes of a single choice cannot both hold. Thus the explanations
`A AND B` and `A AND C` combine as `A AND (B OR C)` without double-counting.

Weights are exact Python fractions. Weighted counting recursively sums each
outcome's weight times its child count. Variables absent from an explanation
integrate to one. Conditional queries compute `weight(Q AND E) / weight(E)`.
False evidence uses diagram complementation after the positive-rule fixed
point. Zero-probability evidence is an explicit error.

Hard mode inserts the selected neural category as observed signed facts. Soft
mode represents one categorical variable per proposition; positive and negative
support are correlated through its four outcomes. Distinct neural variables
are assumed independent, conditional on the supplied text and fixed batch
assessments. Joint encoding alone does not establish this assumption. The
GLiNER adapter currently applies softmax with no calibration.

Ground queries return exact probabilities. Queries with variables additionally
return ground answer bindings and their marginals; the query's overall
probability means that at least one answer exists, not the sum of answer
probabilities. Exported diagrams contain reachable nodes and the evidence
root so the computation can be inspected.

## Local model and cache

The optional GLiNER adapter loads an immutable Hugging Face revision on CPU in
FP32, enables PyTorch deterministic algorithms, and uses a fixed task schema.
It independently compiles each schema to preserve slot order. It checks the
encoded token length and rejects oversized requests instead of silently
truncating or changing the requested grouping.

The decision cache keys the complete world text, ordered candidate definitions,
model revision, prompt version, label definitions, precision, and runtime
versions/configuration. Different schemas or text produce different keys.
Raw score weights are validated and normalized into exact fractions; their
numeric precision does not make them calibrated probabilities. A cache can be
in-memory or stored as atomic JSON records on disk.

Replay requires the same model/runtime and scheduling configuration. Bitwise
equality across arbitrary hardware is not guaranteed. Recorded traces expose
batch membership, cache keys/hits, and the distinction between query-driven
calls and optional completion of all benchmark assessments.

## Limits

This implementation has no function symbols, negation-as-failure, cut,
unrestricted higher-order goals, incremental world updates, or parameter
learning. Its small decision diagram implementation can grow exponentially.
Configured limits bound agenda events, tables, and diagram nodes; hitting one
raises an error rather than returning an incomplete probability. The current
diagram implementation accepts at most 256 categorical variables.

The model interface is synchronous: fair symbolic work exposes a frontier,
then a model call services part of it. This is cooperative scheduling, not
parallel model execution. INT8 inference and larger classifiers remain future
experiments; the first adapter uses the already-tested FP32 small checkpoint.
