# Python DSL design and implementation plan

Status: proposed design. The Python examples describe the intended API; the DSL
is not implemented. The existing engine API remains supported.

Relational Decisions will gain a Python embedded DSL for composing typed
relations, mixing structured facts with natural-language evidence, and inspecting
the resulting answers. Annotated definitions compile into immutable relation
values. Those values support functional composition, explicit argument binding,
and queries with declarative holes.

The MVP builds on the current finite positive-Datalog engine and its exact
probabilistic explanations. A small declarative intermediate representation
supports compilation and inspection. Resumable evaluation, checkpoints, and
branch management are deferred; they are not prerequisites for this DSL.

## Design direction

The agreed direction is Python, functional composition, immutable structures,
typed symbolic parameters, and a restrained amount of syntactic magic.

- Decorated functions define relations through a checked, pure expression
  language. Parameter annotations introduce typed symbolic variables.
- Annotated model fields declare scalar values and relationship cardinality.
  Standard Python type checkers preserve those types through composition, query
  construction, and decoded results; this is an MVP requirement.
- Relation objects are first-class values. Functions and combinators can build
  larger definitions from smaller ones.
- Natural-language statements describe evidence goals. Constructing them does
  not assert a fact or invoke a model.
- Queries bind some arguments and leave others as typed holes over finite domains.
- Evaluation is explicit. Results preserve exact probabilities, authored names,
  evidence assessments, and explanations.

The design borrows typed function wrappers and composition from
[Hask](https://github.com/billpmurphy/hask#the-type-system-and-typed-functions),
ordinary composable data from [Hiccup](https://github.com/weavejester/hiccup),
relational variables from [core.logic](https://github.com/clojure/core.logic),
and the distinction between goals and their construction from
[Refinery](https://github.com/TOTBWF/refinery).
[Snake ORM's descriptor design](https://velezanthony.github.io/snake-orm/users/reference/typing/)
demonstrates different static types for instance and class access.
[Pony's queries](https://docs.ponyorm.org/queries.html) demonstrate translating
ordinary model navigation into relational operations.
[SQLModel's relationship declarations](https://sqlmodel.tiangolo.com/tutorial/relationship-attributes/define-relationships-attributes/)
show how model annotations express related values, collections, and optionality.
These are design references, not proposed dependencies.

## Authoring experience

### Models declare the schema

Use explicitly typed descriptors for the primary schema surface. `@model` creates
an immutable record and uses `dataclass_transform` to describe constructor and
frozen-field behavior to standard checkers. The descriptor supplies the different
types for instance and class access.

```python
@model
class City:
    id: Field[str] = field(key=True)
    name: Field[str] = field(display=True)


@model
class Organization:
    id: Field[str] = field(key=True)
    name: Field[str] = field(display=True)
    city: One[City] = one()


@model
class Person:
    id: Field[str] = field(key=True)
    name: Field[str] = field(display=True)
    employer: One[Organization] = one()
    home_city: One[City] = one()
    affiliations: Many[Organization] = many()
```

The declaration names the relationship as well as its target type. The compiler
can derive `Person.employer` and `Organization.city` from this schema; it does not
need additional function stubs for each field. Standalone relations still serve
concepts that are not model fields, including arbitrary many-to-many predicates.

| Access | Static type and meaning |
| --- | --- |
| `ada.name` | `str`, a concrete immutable field |
| `ada.employer.city.name` | `str`, navigation through concrete records |
| `Person.name` | `Relation[Person, str]`, a field projection |
| `Person.employer` | `Relation[Person, Organization]`, a declared link |
| `Person.affiliations` | `Collection[Person, Organization]`, an explicitly plural link |

Class access builds a description and never loads data. Instance access reads the
supplied immutable record. It performs no implicit model call or database lookup.

### Relations combine facts and reports

Typed constructors declare extensional relations; decorated functions define
derived relations. Both become immutable relation objects. A unary constructor
infers `Predicate[Person]`; a decorated unary definition retains the same domain.

```python
member = predicate(Person, name="member")


@relation
def endorsed(person: Person) -> Goal:
    return reported(
        "The committee endorses {person}.",
        negative="The committee does not endorse {person}.",
        person=person,
    )


@relation
def eligible(person: Person) -> Goal:
    return member(person) & endorsed(person)
```

Within a checked definition, the compiler binds a symbolic `person` with the
declared `Person` schema and interprets the body. Static checkers see `Person`,
including its real declared attributes. This supports the same attribute names
and field types in both ordinary code and compiled definitions. The body is not
executed with a concrete record or an arbitrary proxy.

An annotation-only function with an ellipsis body is not yet promised as the
declaration syntax: standard checkers can reject a non-abstract function with a
missing return. The typed constructor is the initial checked spelling; any
decorated declaration marker must pass the same checker contract before adoption.

The same derived relation can be expressed through point-free composition:

```python
eligible = (member & endorsed).named("eligible")
```

These forms have equivalent logical meaning. Naming adds diagnostic metadata;
it does not create new independent uncertain facts.

Inline language goals are equally valid:

```python
@relation
def eligible(person: Person) -> Goal:
    return member(person) & reported(
        "The committee endorses {person}.",
        negative="The committee does not endorse {person}.",
        person=person,
    )
```

Factoring an identical language goal into a named relation preserves its
proposition identity. A compiler must not turn that refactor into two model
assessments or two independent choices.

### Relational arrows

Schema projections and binary relations support the same arrow-style composition:

```python
works_in = (Person.employer >> Organization.city).named("works_in")
local_city = (works_in & Person.home_city).named("local_city")
```

`works_in(person, city)` introduces an existential organization connecting the
two relations. `local_city` requires both relations to support the same
person-city pair. The arrow describes how relations connect; it does not
restrict queries to a single input-output direction.

| Operation | Meaning | MVP constraint |
| --- | --- | --- |
| `r & s` | Both support the same tuple | Matching arity and domain types |
| `r | s` | Either supports the tuple | Matching arity and domain types |
| `r >> s` | Join through a fresh intermediate variable | Binary relations with matching middle types |
| `r.bind(person=ada)` | Fix named arguments and retain the others | Names and values must match the signature |
| `r.named("name")` | Attach a readable name | Does not alter logical identity |

For equal signatures, point-free composition aligns arguments by position and
uses the left relation's parameter names for the resulting signature. Explicit
decorated definitions handle reordered arguments. We will not infer argument
permutations from coincidentally equal types.

Relation application requires all remaining arguments. Leaving an argument out
is an error; `.bind(...)` is explicit partial application. A supplied hole is a
logical variable, not an omitted argument.

Named partial binding remains useful, but native typing cannot generally remove
an arbitrary subset of keyword parameters from a `ParamSpec`. The initial
compiler checks named `.bind` precisely. Phase 0 must establish a typed binding
form for the supported unary and binary cases, such as explicit left/right
binding; a result that silently becomes `Any` is not an acceptable substitute.

At the goal level, `&` conjoins and `|` combines alternative proofs. Alternatives
collect support rather than choosing the first successful branch. Python `and`,
`or`, `not`, and truth-value coercion are rejected for symbolic goals. General
arrow products, automatic currying, and unrestricted higher-order relations can
follow after the core examples establish a need.

Collections require an explicit operation. For example,
`Person.affiliations.any(Organization.city.eq(paris))` is a `Predicate[Person]`
whose support comes from at least one related organization in Paris. It lowers
to existential relational joins; matching several organizations does not add
independent probabilities. `Person.affiliations.city` is a type error.

Use `.eq(value)` for a typed projection comparison in the constructor API.
Overloading `object.__eq__` with a symbolic result can weaken checker compatibility
and structural equality. Python comparison syntax inside compiled bodies can be
added separately with explicit lowering rules.

### World and query values

An immutable world supplies a finite entity registry, structured facts, and one
text evidence context. Domain registration specifies stable identity and display
fields. Records have immutable scalar fields, to-one links, and tuple-backed
to-many links. The world validates their keys and references before lowering.

```python
paris = City(id="paris", name="Paris")
lab = Organization(id="lab", name="The Lab", city=paris)
ada = Person(
    id="ada", name="Ada", employer=lab, home_city=paris, affiliations=(lab,)
)
world = World(
    entities=(ada, lab, paris),
    facts=(member(ada),),
    text="The committee endorses Ada.",
)

program = Program(eligible)
person = hole(Person, "person")
query = select(person).where(eligible(person))

result = Evaluator(backend=backend).evaluate(program, query, world=world)
```

`Program` collects reachable relation definitions; constructing it performs no
inference. `Evaluator` is a proposed facade around the existing `Engine`.
`backend` is an explicitly configured existing backend or a deterministic test
backend. A ground goal such as `eligible(ada)` is also a valid query.

Hole identity is independent of its display name. Reusing one hole object joins
on the same variable; creating another hole with the same name creates a distinct
variable. Selected holes must be bound by the query body. Variables used in the
body but omitted from the selection are existential.

World facts are ground applications of declared extensional relations to
registered values. They cannot assert a derived relation or silently override a
language assessment. Any future mechanism for supplying reviewed language
assessments will be a separate explicit input.

Named function parameters provide the primary binding syntax inside definitions.
Explicit `hole` and `select` constructors provide a source-independent query API.
Source-compiled lambda query sugar is a later convenience, not an MVP dependency.

## Natural language semantics

`reported(positive, negative=..., **bindings)` declares a pair of assertions to
assess against the world's original text. It asks whether each assertion is
explicitly supported, including paraphrases. It does not ask whether the assertion
is generally plausible, infer entities from text, or feed derived conclusions
back into the evidence.

Both templates are required in the MVP. This makes the negative assertion
reviewable and matches the existing `Candidate` contract. Automatic negation of
arbitrary prose is not reliable enough to supply it implicitly.

Templates and typed bindings remain separate until grounding. Placeholder names
must match the bindings exactly. Initially, templates allow simple named slots;
attribute expressions, indexing, conversions, and format specifications are
rejected. Ordinary f-strings are rejected inside language-goal definitions because
they erase the structured binding too early.

| Assessment | Positive goal | Explicit refutation goal |
| --- | --- | --- |
| supported | Supported | Unsupported |
| refuted | Unsupported | Supported |
| both | Supported | Supported |
| unknown | Unsupported | Unsupported |

`refuted(endorsed(person))` selects the explicit negative channel when the
relation is a transparent alias for a language goal. It is also valid directly
on a `reported(...)` goal. The compiler rejects refutation of arbitrary composite
relations in the MVP: no general negation, complement, or De Morgan rules are
implied. Missing support does not establish explicit refutation. The four-way
assessment belongs to a primitive language proposition; derived relations do not
automatically acquire a four-way classifier result.

Hard mode uses the selected category as signed facts. Soft mode retains the
current four-outcome categorical variable and its assumptions. Scores are
uncalibrated, and distinct proposition variables are treated as independent.
Positive and negative support for one proposition share the same choice.

### Grounding and identity

Finite typed domains constrain all language-goal arguments. For the MVP, compile
candidate tuples from the declared domains with an explicit maximum candidate
count; reject oversized products before model execution. Domain enumeration
registers possible candidates without forcing their assessment. The existing
query-driven scheduler retains lazy evaluation. A later optimization may derive
smaller candidate sets from structural joins.

Entity keys determine logical identity. Display text determines rendering.
The world validates unique keys within each domain and rejects ambiguous display
labels within a type unless the caller supplies distinct labels. Types and labels
must be presented unambiguously in multi-domain language requests.

A language proposition's descriptor includes its contract version, both
templates, normalized binding slots and their domain types, and evidence scope.
Grounding adds stable entity keys. Identical descriptors and bindings within the
same scope share a candidate, including across inline and named forms. Different
paraphrases are not automatically merged; reuse the same definition when identity
matters. Source locations and cosmetic relation names are diagnostic metadata.

The decision cache must still include exact world text, rendered candidate
definitions, the complete ordered batch, and backend/runtime identity. Changing
wording, entity rendering, or request context must not reuse an incompatible
assessment. DSL contract changes require corresponding identity/version changes.

The MVP has one text context per world. Per-document scoping, evidence retrieval,
and citation extraction are separate extensions. Explanations may identify the
world and assessment used; they must not invent supporting text spans that a
backend did not return.

## Purity and compilation

Purity is a requirement of accepted relation definitions. The decorator captures
source and compiles a restricted Python AST without executing the function body.
It introduces typed symbolic bindings and interprets recognized constructs into
immutable expression nodes. Injecting proxy arguments and running arbitrary
Python would not enforce this requirement.

The initial accepted body contains a docstring, immutable local bindings, and a
single return expression. Extensional relations use typed declaration constructors.
Expressions may reference typed parameters, supported constants, declared relation
values, declared model fields, recognized goal constructors, and the DSL operators.
Local names cannot be reassigned. All captured values must be validated immutable
DSL values or registered scalar/domain constants.

Assignments to attributes or containers, loops, comprehensions, imports, arbitrary
calls, undeclared dynamic attribute access, exceptions, I/O, and Python branching
are rejected.
The compiler must not call user-defined descriptors, conversion methods, or
annotation expressions while resolving a definition. Registered types and symbols
are resolved structurally. Unsupported constructs produce a source-located error
before any part of the body is evaluated.

This guarantee covers the DSL body and its accepted dependencies. Importing a
Python module and evaluating decorators or default arguments follows normal
Python semantics; the DSL is not a sandbox for untrusted Python modules. Defaults
and extra decorators on relation definitions are outside the initial subset.

Pure helper functions can eventually become checked expression-building macros.
Calls from a checked body require a checked DSL helper; a decorator that merely
asserts that an arbitrary callback is pure is insufficient. Ordinary host functions
may compose already constructed relation values outside the DSL compiler, under
ordinary Python execution semantics.

Source-backed declarations are the initial ergonomic path. If source cannot be
retrieved reliably, compilation fails with guidance to use explicit relation and
expression constructors. There is no fallback to executing the body. Notebook and
REPL source capture need dedicated support and tests before being promised.

Relation symbols are registered before bodies are linked, allowing references to
later declarations and finite recursive rules. Compilation resolves recursion as
references between definitions, not recursive calls to Python functions. Missing
or ambiguous symbols fail at program compilation. Explicit program registries
bound name resolution; relation discovery must not depend on a mutable process-wide
registry or import order.

### Type guarantees

Static typing is a first-class contract. Both mypy and Pyright must check the
supported schema, expression, query, and result surface without a plugin, generated
schema stubs, caller-side casts, or `Any` escaping into normal user code.
Definition-time validation remains necessary for semantics that static Python
typing does not express, including rule safety, purity, template placeholders,
symbol resolution, entity identity, and resource limits.

The implementation uses three distinct mechanisms:

1. `dataclass_transform` describes typed constructors and frozen records. It does
   not itself synthesize a symbolic model type or implement immutability at runtime.
2. Overloaded descriptor access returns concrete `T` on an instance and a
   `Relation[Owner, T]` on a class. The owner is inferred from the descriptor's
   `owner: type[Owner]` argument. To-many descriptors return a separate collection
   expression. This retains both ends of an arrow without duplicating the model.
3. Generics and overloads retain relation arguments, hole types, and selected
   result shapes. A `Query[tuple[Person, City]]` evaluates to a
   `Result[tuple[Person, City]]`, not untyped row dictionaries.

This follows the separation in the
[Python typing specification](https://typing.python.org/en/latest/spec/dataclasses.html)
between statically described dataclass behavior and the library's runtime work.
Invariance is the conservative starting point for relation and hole types;
composition must not widen incompatible domains to `object` or a union.

The following are release gates, not optional editor enhancements:

| Contract | Examples checked by both tools |
| --- | --- |
| Construction and immutability | Wrong field types, missing required fields, forbidden mutation |
| Declared navigation | Concrete chained field types; misspelled attributes rejected |
| Relation composition | Exact source/target types; incompatible intermediate types rejected |
| Collections | Explicit existential predicates; scalar navigation on a collection rejected |
| Calls and holes | Wrong entity and hole domain rejected |
| Query results | Selected scalar, entity, and tuple shapes retained through evaluation |

Supported nullable relationships must keep absence in their types and require an
explicit narrowing or optional-path operation. SQL-style null, missing facts, and
four-way language `unknown` are separate concepts. Required/optional/to-many
cardinality is a constraint on supplied structured data; it is not a declaration
that an arbitrary derived or uncertain relation is functional. Optional navigation
must remain inside its logical branch so an absent link cannot remove a match
supported by another branch of an OR.

### Navigation syntax and its limits

The initial symbolic surface favors `Person.employer >> Organization.city`, which
retains both source and target types. A schema-annotated parameter can use ordinary
paths such as `person.employer.city`; the compiler turns those declared fields into
joins without executing user descriptors. No arbitrary `Proxy[Person]` is assumed
to acquire `Person`'s members through Python generics.

Snake ORM's deeper class spelling works by returning `type[Target]` from a
relationship descriptor. Its documentation also notes that such a path appears
callable to the type checker. For this DSL, that representation additionally needs
an explicit way to retain the source domain for typed arrows and distinguish two
aliases of the same model. Prototype deep dotted class navigation separately;
adopt it only if those semantics are explicit and the checker tests pass. Arrows
already provide a compact, typed spelling for the MVP.

Pony demonstrates the ergonomics of generator and lambda queries. Such syntax is
a possible compiled frontend to the same expressions. It does not justify executing
arbitrary generators or permitting side effects in checked definitions. Whole-model
generic proxies, automatic schema generation, and arbitrary partial-parameter
transformations are not consequences of `dataclass_transform`.

A small signature-only feasibility check passed mypy 2.4.0 and Pyright 1.1.414
for constructors, frozen descriptors, concrete nested navigation, owner-preserving
arrows, collection predicates, unary decorated relations, typed holes, and tuple
results. This checks the proposed type shapes, not an implemented DSL. Full
decorator signatures, parameter names, optional links, forward references, and
runtime behavior still need the Phase 0 contract suite and subsequent implementation.

## Declarative representation and engine integration

The DSL uses a small immutable expression representation with source metadata:

| Node | Purpose |
| --- | --- |
| `ModelSchema` and `FieldRef` | Typed fields, cardinality, identity, and source metadata |
| `RelationDef` and `RelationRef` | Typed declarations, names, and recursive references |
| `Var` and `Const` | Scoped typed terms and encoded domain values |
| `Call` | Apply a relation to terms |
| `All` and `Any` | Conjunction and alternative support |
| `Exists` | Scoped intermediate variables |
| `Report` and `RefutedReport` | Bound templates and a selected evidence channel |
| `Query` | Selected variables and a goal expression |

Point-free operators elaborate into this representation. Binding substitutes
constants without mutating a definition. Composite relations retain references
and use fresh variables at each application. Compiler keys normalize variable
identity independently of display names while retaining authored order.

Lowering produces the current engine program dictionaries, candidate registry,
and source/domain maps:

- Model fields and links become typed extensional relations. Scalar fields use
  registered encodings; collection predicates introduce scoped existential joins.
  Required links, key integrity, and declared cardinality are validated before
  evaluation. Repeated use of the same field path shares the intended variable;
  explicit aliases and separate existential binders remain distinct.
- Conjunction becomes rule bodies. Alternatives become multiple rules with a
  shared head; helper predicates avoid exponential distributive expansion.
- Arrow composition introduces a fresh body variable. Explicit domain predicates
  are available for finite enumeration where needed. Unsafe or unbounded variables
  are rejected rather than guessed.
- Domain values encode to collision-free, type-qualified string constants.
  Decoding maps answers back to registered immutable Python values.
- Reports become `Candidate` entries with paired signed atoms. All occurrences of
  one grounded proposition use the same owner and categorical choice.
- Selected queries lower through a generated query relation so bindings and
  existential projection map back consistently.

The integration seam is additive. Keep `Engine(program, candidates=..., world=...)`,
the benchmark format, and existing model adapters compatible. Current relevant
modules are [engine.py](../src/relational_decisions/engine.py),
[terms.py](../src/relational_decisions/terms.py),
[decisions.py](../src/relational_decisions/decisions.py), and
[prompts.py](../src/relational_decisions/prompts.py). The supported semantics and
resource limits remain those in [the runtime design](engine.md).

### Algebra and execution context

Conjunction, alternative support, and composition obey their relational meanings
for fixed primitive assessments. Shared proofs must not be counted as independent
draws. In particular, `p & p` and `p | p` preserve `p`'s support and probability.

Some existing models change their answers when request grouping changes. Therefore
logical equivalence does not promise identical model outputs under different
execution plans. The compiler preserves authored operand order and emits a
deterministic plan; it does not freely reorder model-bearing expressions for
optimization. Record compiler version, plan fingerprint, and scheduler settings.
Algebra tests use fixed assessments; live-model comparisons also control request
context. Inline and named extraction must preserve candidate identity, but identical
batch schedules across every equivalent refactor are not an MVP guarantee.

## Results and diagnostics

A result facade exposes decoded bindings, exact query and answer probabilities,
primitive assessments, execution traces, and explanation access. For a query with
variables, the overall probability means that at least one answer exists; it is
not the sum of answer marginals.

Diagnostics refer to authored relations and source locations. Examples include
an incompatible middle type in `Person.employer >> Person.home_city`, an unbound
template slot, an unsafe selected variable, and an unsupported call inside a pure definition.
Generated predicate IDs remain available for debugging without dominating the
ordinary display.

The MVP runs to completion through the existing engine. A resource-limit error
must not become a completed result with probability zero. An unassessed candidate
must not be displayed as an assessed `unknown`. Explanation rendering must show
shared support without presenting overlapping proofs as additive probabilities.

## Phased implementation

Phases are sequential and each ends with a reviewable example and acceptance
checks. Phase 1 provides a usable constructor API; later phases add the preferred
authoring syntax and language integration. No phase requires checkpointing work.

### Phase 0 Type contracts and semantic examples

Build signature-level positive and negative fixtures under pinned mypy and Pyright
versions before committing to the runtime surface. Include typed descriptor
constructors, frozen values, concrete field paths, owner-preserving projections,
collections, optional links, decorated unary/binary signatures, forward references,
holes, typed partial binding, and selected result shapes. Ship inline typing with
the library; the feasibility fixtures are not generated per-model user stubs.

Finalize semantic examples for membership and endorsement, the
person-organization-city join, a recursive reachability relation, and overlapping
uncertain proofs. Confirm template pairs, binding syntax, hole scope, and domain
codecs. Specify behavior for same-model aliases and optional paths under OR.

Acceptance: both checkers accept the positive cases with exact asserted types and
reject every marked invalid case at its intended line. Unknown types, `Any`, and
caller casts cannot make a positive case pass. Each semantic example also has an
expected engine-level translation or diagnostic. A syntax that loses a required
static guarantee is revised before runtime implementation.

### Phase 1 Immutable expressions and engine bridge

Add a `relational_decisions.dsl` package containing domain/signature types,
schema descriptors and immutable records, expression values, explicit constructors,
queries, and the lowering pass. Begin with structured facts, `All`, `Any`, scoped
variables, and named recursive
references. Support legacy categorical choices in differential fixtures without
designing new choice syntax yet. Keep the existing public engine API intact.

Acceptance: structured joins and recursive examples run through the current
engine with decoded answers. Differential checks against equivalent direct engine
programs preserve exact probabilities, including shared uncertain facts and
existential query semantics. Invalid arity, domain values, and unsafe variables
fail before evaluation. Schema projections agree with equivalent explicit facts
and joins. Optional links and to-many witnesses preserve branch and probability
semantics. Runtime frozen behavior matches its type contract. Composition does
not mutate reused definitions, and Phase 0 checker fixtures continue to pass.

### Phase 2 Annotated relation definitions

Implement source capture, the restricted AST compiler, declaration/linking scopes,
and source-located diagnostics. Compile typed function parameters and immutable
locals and schema field paths without executing the body. Add the constructor
fallback for missing source and supported checked helper expansion. Preserve
callable argument types in the decorated relation wrappers.

Acceptance: decorated and constructor versions of the structured examples agree.
Forward references and recursion link deterministically. A definition containing
an observable side effect is rejected without performing the effect. Mutable
captures, unsupported annotations, Python boolean operators, and unavailable
source produce actionable diagnostics. Fresh variables do not leak across calls,
and two aliases of one schema do not collapse. Typed constructor, descriptor, and
decorator signatures continue to pass both static checkers.

### Phase 3 Natural language goals

Implement paired report templates, typed grounding, finite candidate enumeration
with a size limit, exact descriptor deduplication, refutation selection, and cache
integration. Connect existing backend adapters through the current decision
interface. Definition and compilation must work without ML dependencies.

Acceptance: deterministic fixtures cover supported, refuted, both, and unknown in
hard mode and exact supplied distributions in soft mode. Inline and extracted
reports share candidate identity. Repeated uses make one assessment per run, and
positive/negative channels share one choice. Changes to templates, bindings,
rendering, text, or ordered batch invalidate incompatible cache entries. An
oversized candidate product fails before model work. Tests need no model download.

### Phase 4 Functional composition

Expose typed relation-level `&`, `|`, binary `>>`, explicit `.bind`, and naming.
Preserve readable expression representations and source provenance through
composition. Implement the signatures established in Phase 0, including a statically
typed partial-binding form for supported arities. Keep semantic validation in the
compiler and test it alongside static typing.

Acceptance: point-free and decorated examples agree under fixed assessments.
Check joins with multiple intermediate witnesses, repeated variables, partial
binding, and reuse of the same relation under different names. Conjunction and
union retain shared support and obey idempotence; incompatible compositions fail
in both checkers, with runtime/compiler diagnostics for unchecked callers.
Deterministic lowering and batch-order regression checks cover model-bearing examples.

### Phase 5 Evaluation facade and MVP release

Finish immutable `World` and `Program` values, the evaluator facade, decoded result
objects with preserved generic result shapes, and explanation rendering. Publish
a getting-started example that combines structured membership with textual
endorsement and displays an exact answer plus its supporting assessments.
Use an explicit deterministic backend for a fast,
reproducible example, and document how to select an existing local model backend.

Acceptance: a user can install the core package, declare domain types and relations,
compose a rule, supply facts and text, ask a ground or open query, and inspect its
answer without editing benchmark JSON. The user-facing examples pass mypy and
Pyright without casts, suppressions, or plugins. Existing engine, adapter, and
benchmark checks still pass. One documented optional model smoke run verifies
the adapter boundary; model accuracy is not a DSL correctness gate. Errors and
examples clearly distinguish missing support, explicit refutation, and unevaluated work.

## Scope and remaining choices

The MVP includes the checked decorator surface, immutable expressions, finite typed
model schemas with fields and relationships, explicit holes, natural-language
reports, three relation operators, explicit partial binding, and inspectable
evaluation through the existing engine.
It preserves finite positive recursion and the existing probability semantics.

New DSL syntax for declaring structured categorical choices is deferred. Existing
engine programs retain that capability; DSL users initially obtain probabilistic
results through language assessments in soft mode. Exactness describes inference
given supplied weights, not the accuracy or calibration of those weights.

Deferred work includes checkpointing and branches, incremental world updates,
program synthesis and authoring holes, SMT verification, unrestricted miniKanren,
linear resource consumption, general negation, arbitrary Python execution inside
definitions, automatic entity extraction, learned calibration, and multi-document
retrieval. Rosette, Refinery, and linear logic remain useful future references;
their capabilities are not implied by the MVP syntax.

Before the relevant implementation phase, settle these narrower choices:

| Choice | Proposed starting point | Decision point |
| --- | --- | --- |
| Public module and constructor names | `relational_decisions.dsl`; names shown here | Phase 0 |
| Schema declarations | Frozen typed descriptor models with key and display metadata | Phase 0 |
| Optional paths and aliases | Explicit typed operations with branch-local joins | Phase 0 |
| Partial binding | Typed unary/binary form plus compiler-checked named binding | Phase 0 |
| Recursive definition scope | Explicit program symbol registry and deferred linking | Phase 2 |
| Checked helper syntax | A small marked helper form using the same accepted AST | Phase 2 |
| Candidate enumeration budget | Explicit configurable bound with a conservative default | Phase 3 |
| Notebook source support | Constructor API first; source capture only when reliable | After Phase 2 |
| Static checker contract | Both mypy and Pyright; no plugin, generated schema stubs, or public `Any` | Phase 0 and every phase |
| Display format | Concise bindings and probabilities, expandable evidence details | Phase 5 |

The design succeeds when a useful relation takes a few readable lines, composes
without knowledge of its implementation, and still expands into understandable
rules and evidence. That is the criterion for adding further syntax.
