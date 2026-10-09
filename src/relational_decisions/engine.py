"""Fair continuation scheduling with variant tabling and symbolic provenance.

Tables contain ground answers and canonical decision-diagram supports. A new
or enlarged support wakes subscribers; idempotent union reaches the least
fixed point even for left recursion. Ground language goals suspend until a
shared batch is assessed. This is a finite positive-Datalog engine, not a full
implementation of the miniKanren language.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from fractions import Fraction

from .decisions import LABELS, Candidate, DecisionCache, best_label
from .diagram import Diagram, ResourceLimit
from .terms import Terms, Var, ground, reify, unify, variant


@dataclass(frozen=True)
class EngineConfig:
    schedule: str = "frontier"
    mode: str = "hard"
    batch_size: int = 4
    search_quantum: int = 64
    max_steps: int = 200_000
    max_nodes: int = 100_000
    max_tables: int = 20_000

    def __post_init__(self):
        if self.schedule not in ("frontier", "full", "single") or self.mode not in (
            "hard",
            "soft",
        ):
            raise ValueError("Unknown scheduler or assessment mode")
        if (
            min(
                self.batch_size,
                self.search_quantum,
                self.max_steps,
                self.max_nodes,
                self.max_tables,
            )
            < 1
        ):
            raise ValueError("Budgets must be positive")


@dataclass
class Table:
    goal: tuple
    answers: dict = field(default_factory=dict)
    subscribers: list = field(default_factory=list)


@dataclass
class Continuation:
    owner: int
    goals: tuple
    subst: dict
    support: int


@dataclass
class InferenceResult:
    query_probabilities: dict
    evidence_probability: Fraction
    answers: dict
    assessments: dict
    distributions: dict
    trace: list
    stats: dict
    explanations: dict

    def to_dict(self):
        return {
            "query_probabilities": {
                k: str(v) for k, v in self.query_probabilities.items()
            },
            "evidence_probability": str(self.evidence_probability),
            "answers": self.answers,
            "assessments": self.assessments,
            "distributions": {
                k: {label: str(p) for label, p in v.items()}
                for k, v in self.distributions.items()
            },
            "trace": self.trace,
            "stats": self.stats,
            "explanations": self.explanations,
        }


class Engine:
    def __init__(
        self, program, *, candidates=(), world="", backend=None, config=None, cache=None
    ):
        self.program = program
        self.world = world
        self.backend = backend
        self.config = config or EngineConfig()
        self.cache = cache if cache is not None else DecisionCache()
        self.terms = Terms()
        self.diagram = Diagram(self.config.max_nodes)
        self.tables = []
        self.table_ids = {}
        self.queue = deque()
        self.seen_states = set()
        self.pending = set()
        self.neural_waiters = defaultdict(list)
        self.distributions = {}
        self.trace = []
        self._used = False
        self.steps = self.answer_updates = self.peak_queue = 0
        self.base = defaultdict(dict)
        self.rules = defaultdict(list)
        self.neural_index = defaultdict(list)
        self.neural_variables = {}
        self.arities = {}
        self.candidates = {}
        for candidate in candidates:
            c = (
                Candidate.from_dict(candidate)
                if isinstance(candidate, dict)
                else candidate
            )
            if c.id in self.candidates:
                raise ValueError(f"Duplicate candidate ID: {c.id}")
            self.candidates[c.id] = c
        self._prepare()

    @staticmethod
    def _signature(atom):
        return (atom[0], len(atom))

    def _parse(self, raw, variables=None, require_ground=False):
        atom = self.terms.parse(raw, variables)
        arity = len(atom) - 1
        if atom[0] in self.arities and self.arities[atom[0]] != arity:
            raise ValueError(f"Inconsistent arity for {atom[0]}")
        self.arities[atom[0]] = arity
        if require_ground and not ground(atom):
            raise ValueError(
                "Facts, candidates, choice outcomes, and evidence must be ground"
            )
        return atom

    def _add_base(self, atom, support):
        bucket = self.base[self._signature(atom)]
        bucket[atom] = self.diagram.or_(bucket.get(atom, 0), support)

    def _prepare(self):
        for fact in self.program.get("facts", []):
            self._add_base(self._parse(fact, require_ground=True), 1)
        choice_ids = set()
        for choice in sorted(self.program.get("choices", []), key=lambda c: c["id"]):
            if choice["id"] in choice_ids:
                raise ValueError("Duplicate probabilistic choice ID")
            choice_ids.add(choice["id"])
            outcomes = choice["outcomes"]
            variable = self.diagram.register(
                "program:" + choice["id"],
                tuple(str(i) for i in range(len(outcomes))),
                [o["p"] for o in outcomes],
            )
            for index, outcome in enumerate(outcomes):
                support = self.diagram.event(variable, [index])
                for fact in outcome["facts"]:
                    self._add_base(self._parse(fact, require_ground=True), support)
        for rule in self.program.get("rules", []):
            variables = {}
            head = self._parse(rule["head"], variables)
            body = tuple(self._parse(a, variables) for a in rule["body"])
            if not body:
                raise ValueError("Use facts for empty-body rules")
            head_vars = {t for t in head if isinstance(t, Var)}
            body_vars = {t for a in body for t in a if isinstance(t, Var)}
            if not head_vars <= body_vars:
                raise ValueError("Head variables must occur in the body")
            self.rules[self._signature(head)].append(rule)
        owners = {}
        for cid, candidate in sorted(self.candidates.items()):
            for positive, raw in (
                (True, candidate.atom),
                (False, candidate.negative_atom),
            ):
                atom = self._parse(raw, require_ground=True)
                if atom in owners:
                    raise ValueError(
                        "Each ground neural atom needs one canonical candidate owner"
                    )
                owners[atom] = cid
                self.neural_index[self._signature(atom)].append((cid, atom, positive))
            if self.config.mode == "soft":
                self.neural_variables[cid] = self.diagram.register(
                    "neural:" + cid, LABELS
                )

    def _table(self, goal):
        key = variant([goal])
        if key not in self.table_ids:
            if len(self.tables) >= self.config.max_tables:
                raise ResourceLimit("Goal table budget exhausted")
            index = len(self.tables)
            self.table_ids[key] = index
            self.tables.append(Table(goal))
            self.queue.append(("expand", index))
        return self.table_ids[key]

    def _enqueue(self, continuation):
        if continuation.support == 0:
            return
        owner_goal = reify(self.tables[continuation.owner].goal, continuation.subst)
        goals = tuple(reify(g, continuation.subst) for g in continuation.goals)
        key = (continuation.owner, variant((owner_goal,) + goals), continuation.support)
        if key not in self.seen_states:
            self.seen_states.add(key)
            self.queue.append(("resume", continuation))

    def _publish(self, table_id, atom, support):
        if not ground(atom):
            raise ValueError("A safe Datalog rule produced a nonground answer")
        table = self.tables[table_id]
        old = table.answers.get(atom, 0)
        new = self.diagram.or_(old, support)
        if new == old:
            return
        table.answers[atom] = new
        self.answer_updates += 1
        # Wakeups are queued to keep cyclic propagation off the Python stack.
        for continuation in table.subscribers:
            self.queue.append(("deliver", (continuation, atom, new)))

    def _deliver(self, continuation, atom, support):
        subst = unify(continuation.goals[0], atom, continuation.subst)
        if subst is not None:
            self._enqueue(
                Continuation(
                    continuation.owner,
                    continuation.goals[1:],
                    subst,
                    self.diagram.and_(continuation.support, support),
                )
            )

    def _expand(self, table_id):
        table = self.tables[table_id]
        signature = self._signature(table.goal)
        for atom, support in self.base.get(signature, {}).items():
            if unify(table.goal, atom, {}) is not None:
                self._publish(table_id, atom, support)
        for cid, atom, positive in self.neural_index.get(signature, []):
            if unify(table.goal, atom, {}) is None:
                continue
            if cid in self.distributions:
                self._publish(table_id, atom, self._neural_support(cid, positive))
            else:
                self.pending.add(cid)
                self.neural_waiters[cid].append((table_id, atom, positive))
        for raw in self.rules.get(signature, []):
            variables = {}
            head = self.terms.parse(raw["head"], variables)
            body = tuple(self.terms.parse(a, variables) for a in raw["body"])
            subst = unify(table.goal, head, {})
            if subst is not None:
                self._enqueue(Continuation(table_id, body, subst, 1))

    def _resume(self, continuation):
        if not continuation.goals:
            atom = reify(self.tables[continuation.owner].goal, continuation.subst)
            self._publish(continuation.owner, atom, continuation.support)
            return
        goal = reify(continuation.goals[0], continuation.subst)
        table = self.tables[self._table(goal)]
        table.subscribers.append(continuation)
        for atom, support in table.answers.items():
            self.queue.append(("deliver", (continuation, atom, support)))

    def _neural_support(self, cid, positive):
        allowed = ("supported", "both") if positive else ("refuted", "both")
        if self.config.mode == "hard":
            return int(best_label(self.distributions[cid]) in allowed)
        return self.diagram.event(
            self.neural_variables[cid], [LABELS.index(x) for x in allowed]
        )

    def _flush(self, purpose):
        if self.backend is None:
            raise ValueError(
                "A decision backend is required for requested neural predicates"
            )
        if self.config.schedule == "full":
            ids = sorted(set(self.candidates) - set(self.distributions))
        else:
            size = 1 if self.config.schedule == "single" else self.config.batch_size
            ids = sorted(self.pending)[:size]
        requests = tuple(self.candidates[cid] for cid in ids)
        start = time.perf_counter()
        assessments, key, hit = self.cache.evaluate(self.backend, self.world, requests)
        self.trace.append(
            {
                "candidate_ids": ids,
                "cache_key": key,
                "cache_hit": hit,
                "purpose": purpose,
                "seconds": time.perf_counter() - start,
            }
        )
        for cid in ids:
            self.distributions[cid] = assessments[cid]
            self.pending.discard(cid)
            if self.config.mode == "soft":
                self.diagram.set_weights(
                    self.neural_variables[cid],
                    [assessments[cid][label] for label in LABELS],
                )
            for table_id, atom, positive in self.neural_waiters.pop(cid, []):
                self._publish(table_id, atom, self._neural_support(cid, positive))

    def _drain(self, purpose="inference"):
        while self.queue or self.pending:
            for _ in range(self.config.search_quantum):
                if not self.queue:
                    break
                self.peak_queue = max(self.peak_queue, len(self.queue))
                self.steps += 1
                if self.steps > self.config.max_steps:
                    raise ResourceLimit("Search step budget exhausted")
                kind, value = self.queue.popleft()
                if kind == "expand":
                    self._expand(value)
                elif kind == "resume":
                    self._resume(value)
                else:
                    self._deliver(*value)
            if self.pending:
                self._flush(purpose)

    def run(self, *, complete_assessments=False):
        if self._used:
            raise ValueError("Use a fresh Engine for each world snapshot/run")
        self._used = True
        start = time.perf_counter()
        query_tables = {}
        for query in self.program.get("queries", []):
            if query["id"] in query_tables:
                raise ValueError("Duplicate query ID")
            query_tables[query["id"]] = self._table(self._parse(query["atom"]))
        evidence_tables = []
        for e in self.program.get("evidence", []):
            if not isinstance(e["value"], bool):
                raise TypeError("Evidence values must be Boolean")
            evidence_tables.append(
                (self._table(self._parse(e["atom"], require_ground=True)), e["value"])
            )
        self._drain()
        inference_assessments = len(self.distributions)
        if complete_assessments:
            self.pending.update(set(self.candidates) - set(self.distributions))
            self._drain("complete_assessments")

        def table_support(table_id):
            result = 0
            for node in self.tables[table_id].answers.values():
                result = self.diagram.or_(result, node)
            return result

        evidence = 1
        for table_id, positive in evidence_tables:
            node = table_support(table_id)
            evidence = self.diagram.and_(
                evidence, node if positive else self.diagram.not_(node)
            )
        count = self.diagram.counter()
        denominator = count(evidence)
        if not denominator:
            raise ValueError("Conditioning evidence has zero probability")
        probabilities, answers, roots = {}, {}, {"evidence": evidence}
        for qid, table_id in query_tables.items():
            support = table_support(table_id)
            numerator = self.diagram.and_(support, evidence)
            probabilities[qid] = count(numerator) / denominator
            roots["query:" + qid] = numerator
            answers[qid] = [
                {
                    "atom": list(atom),
                    "probability": str(
                        count(self.diagram.and_(node, evidence)) / denominator
                    ),
                }
                for atom, node in sorted(self.tables[table_id].answers.items())
            ]
        stats = {
            "steps": self.steps,
            "tables": len(self.tables),
            "diagram_nodes": len(self.diagram.nodes),
            "answer_updates": self.answer_updates,
            "peak_queue": self.peak_queue,
            "batches": len(self.trace),
            "backend_calls": sum(not t["cache_hit"] for t in self.trace),
            "inference_assessments": inference_assessments,
            "assessments": len(self.distributions),
            "seconds": time.perf_counter() - start,
        }
        return InferenceResult(
            probabilities,
            denominator,
            answers,
            {cid: best_label(d) for cid, d in self.distributions.items()},
            self.distributions,
            self.trace,
            stats,
            self.diagram.export(roots),
        )
