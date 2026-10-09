import importlib.util
import json
import random
import tempfile
import unittest
from fractions import Fraction as F
from pathlib import Path

from relational_decisions import (
    Candidate,
    DecisionCache,
    Engine,
    EngineConfig,
    OracleBackend,
)
from relational_decisions.diagram import Diagram, ResourceLimit

ROOT = Path(__file__).resolve().parents[1]


def rule(head, *body):
    return {"head": head, "body": list(body)}


def choice(name, p, atom=None):
    return {
        "id": name,
        "outcomes": [
            {"p": str(p), "facts": [atom or [name]]},
            {"p": str(1 - F(p)), "facts": []},
        ],
    }


def program(*rules, facts=(), choices=(), queries=None, evidence=()):
    return {
        "facts": list(facts),
        "rules": list(rules),
        "choices": list(choices),
        "queries": queries or [{"id": "q", "atom": ["q"]}],
        "evidence": list(evidence),
    }


def candidate(name):
    return Candidate(
        name, (name,), ("not_" + name,), name + " is reported", name + " is denied"
    )


class FixedBackend:
    def __init__(self, outputs):
        self.outputs = outputs
        self.calls = []

    @property
    def identity(self):
        return {"backend": "test-fixed", "outputs": self.outputs}

    def assess(self, world, candidates):
        self.calls.append([c.id for c in candidates])
        return {c.id: self.outputs[c.id] for c in candidates}


class DiagramTests(unittest.TestCase):
    def test_shared_diamond_and_repeated_fact(self):
        d = Diagram()
        a, b, c = [
            d.event(d.register(name, ["no", "yes"], [1 - p, p]), [1])
            for name, p in [("a", F(1, 5)), ("b", F(3, 10)), ("c", F(1, 2))]
        ]
        diamond = d.or_(d.and_(a, b), d.and_(a, c))
        self.assertEqual(d.counter()(diamond), F(13, 100))
        self.assertEqual(d.and_(a, a), a)
        self.assertEqual(diamond, d.and_(a, d.or_(b, c)))
        self.assertEqual(d.or_(a, d.not_(a)), 1)

    def test_categorical_exclusivity(self):
        d = Diagram()
        v = d.register("mode", ["a", "b", "c"], [F(1, 4), F(1, 4), F(1, 2)])
        self.assertEqual(d.and_(d.event(v, [0]), d.event(v, [1])), 0)
        self.assertEqual(d.counter()(d.event(v, [0, 1])), F(1, 2))


class EngineTests(unittest.TestCase):
    def test_nonlinear_recursion_against_exhaustive_reference(self):
        # Exercise compositions beyond the fixed benchmark templates: repeated
        # recursive subgoals, multi-fact outcomes, joins, and cyclic propagation.
        spec = importlib.util.spec_from_file_location(
            "reference_logic", ROOT / "benchmarks/nl_logic_v1/logic.py"
        )
        reference = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference)
        rng = random.Random(31009)
        nodes = ["a", "b", "c", "d"]
        for case in range(40):
            choices = []
            for i in range(3):
                edges = [
                    ["edge", rng.choice(nodes), rng.choice(nodes)] for _ in range(2)
                ]
                choices.append(
                    {
                        "id": str(i),
                        "outcomes": [
                            {"p": "1/3", "facts": edges},
                            {"p": "2/3", "facts": []},
                        ],
                    }
                )
            p = program(
                rule(
                    ["reach", "?x", "?y"], ["reach", "?x", "?z"], ["reach", "?z", "?y"]
                ),
                rule(["reach", "?x", "?y"], ["edge", "?x", "?y"]),
                rule(["selected", "?x"], ["reach", "?x", "?y"], ["wanted", "?y"]),
                rule(["selected", "?x"], ["selected", "?y"], ["edge", "?y", "?x"]),
                facts=[["wanted", rng.choice(nodes)]],
                choices=choices,
                queries=[{"id": x, "atom": ["selected", x]} for x in nodes],
            )
            with self.subTest(case=case):
                self.assertEqual(
                    Engine(p).run().query_probabilities,
                    {
                        k: F(v)
                        for k, v in reference.solve(p, [])[
                            "query_probabilities"
                        ].items()
                    },
                )

    def test_all_open_ground_truths_in_both_modes(self):
        # Uses the frozen independent exhaustive-world oracle's outputs.
        for split in ("train", "dev"):
            path = ROOT / "benchmarks/nl_logic_v1/data"
            gold = {
                r["id"]: r
                for r in map(
                    json.loads, (path / f"{split}.gold.jsonl").read_text().splitlines()
                )
            }
            for inp in map(
                json.loads, (path / f"{split}.inputs.jsonl").read_text().splitlines()
            ):
                target = gold[inp["id"]]
                for mode in ("hard", "soft"):
                    with self.subTest(split=split, id=inp["id"], mode=mode):
                        result = Engine(
                            inp["program"],
                            candidates=inp["candidates"],
                            world=inp["world_text"],
                            backend=OracleBackend(target["assessments"]),
                            config=EngineConfig(mode=mode),
                        ).run(complete_assessments=True)
                        self.assertEqual(
                            result.query_probabilities,
                            {k: F(v) for k, v in target["query_probabilities"].items()},
                        )
                        self.assertEqual(
                            result.evidence_probability,
                            F(target["evidence_probability"]),
                        )
                        self.assertEqual(result.assessments, target["assessments"])

    def test_left_recursion_enumerates_bindings_and_respects_aliases(self):
        p = program(
            rule(["reach", "?x", "?y"], ["reach", "?x", "?z"], ["edge", "?z", "?y"]),
            rule(["reach", "?x", "?y"], ["edge", "?x", "?y"]),
            facts=[["edge", "a", "b"], ["edge", "b", "c"], ["edge", "c", "a"]],
            queries=[
                {"id": "from_a", "atom": ["reach", "a", "?who"]},
                {"id": "loops", "atom": ["reach", "?x", "?x"]},
            ],
        )
        result = Engine(p).run()
        self.assertEqual(
            {tuple(a["atom"]) for a in result.answers["from_a"]},
            {("reach", "a", x) for x in "abc"},
        )
        self.assertEqual(
            {tuple(a["atom"]) for a in result.answers["loops"]},
            {("reach", x, x) for x in "abc"},
        )

    def test_pure_cycle_and_later_successful_branch(self):
        p = program(rule(["q"], ["loop"]), rule(["loop"], ["loop"]), rule(["q"], ["a"]))
        backend = OracleBackend({"a": "supported"})
        result = Engine(
            p,
            candidates=[candidate("a")],
            backend=backend,
            config=EngineConfig(search_quantum=1),
        ).run()
        self.assertEqual(result.query_probabilities["q"], 1)
        self.assertLess(result.stats["steps"], 30)

    def test_frontier_batches_ready_branches_and_stages_later_goals(self):
        p = program(rule(["q"], ["a"], ["c"]), rule(["q"], ["b"]))
        backend = OracleBackend({x: "supported" for x in "abc"})
        result = Engine(
            p, candidates=list(map(candidate, "abc")), backend=backend
        ).run()
        self.assertEqual(
            [t["candidate_ids"] for t in result.trace], [["a", "b"], ["c"]]
        )
        self.assertEqual(result.query_probabilities["q"], 1)

    def test_false_branch_does_not_request_later_neural_goal(self):
        p = program(rule(["q"], ["a"], ["b"]))
        result = Engine(
            p,
            candidates=list(map(candidate, "ab")),
            backend=OracleBackend({"a": "unknown", "b": "supported"}),
        ).run()
        self.assertEqual(result.query_probabilities["q"], 0)
        self.assertEqual(set(result.assessments), {"a"})

    def test_shared_neural_choice_and_correlated_positive_negative(self):
        weights = {
            "supported": "1/2",
            "refuted": "1/5",
            "both": "1/10",
            "unknown": "1/5",
        }
        p = program(
            rule(["q"], ["a"], ["a"]),
            rule(["conflict"], ["a"], ["not_a"]),
            queries=[
                {"id": "q", "atom": ["q"]},
                {"id": "conflict", "atom": ["conflict"]},
            ],
        )
        backend = FixedBackend({"a": weights})
        result = Engine(
            p,
            candidates=[candidate("a")],
            backend=backend,
            config=EngineConfig(mode="soft"),
        ).run()
        self.assertEqual(
            result.query_probabilities, {"q": F(3, 5), "conflict": F(1, 10)}
        )
        self.assertEqual(backend.calls, [["a"]])

    def test_conditioning_on_false_evidence(self):
        p = program(
            rule(["q"], ["a"]),
            choices=[choice("a", F(1, 5)), choice("b", F(3, 10))],
            evidence=[{"atom": ["b"], "value": False}],
        )
        result = Engine(p).run()
        self.assertEqual(result.query_probabilities["q"], F(1, 5))
        self.assertEqual(result.evidence_probability, F(7, 10))
        with self.assertRaisesRegex(ValueError, "zero probability"):
            Engine(
                program(facts=[["b"]], evidence=[{"atom": ["b"], "value": False}])
            ).run()

    def test_rule_order_and_scheduling_preserve_oracle_semantics(self):
        rules = [rule(["q"], ["a"], ["gate"]), rule(["q"], ["b"], ["gate"])]
        for order in (rules, list(reversed(rules))):
            for schedule in ("frontier", "full", "single"):
                for quantum in (1, 64):
                    p = program(*order, choices=[choice("gate", F(3, 10))])
                    result = Engine(
                        p,
                        candidates=list(map(candidate, "ab")),
                        backend=OracleBackend({"a": "supported", "b": "supported"}),
                        config=EngineConfig(schedule=schedule, search_quantum=quantum),
                    ).run()
                    self.assertEqual(result.query_probabilities["q"], F(3, 10))

    def test_budget_and_invalid_program_errors_are_explicit(self):
        with self.assertRaises(ResourceLimit):
            Engine(
                program(rule(["q"], ["a"]), facts=[["a"]]),
                config=EngineConfig(max_steps=1),
            ).run()
        with self.assertRaisesRegex(ValueError, "Head variables"):
            Engine(program(rule(["q", "?x"], ["a"])))
        with self.assertRaisesRegex(ValueError, "sum exactly"):
            Engine(
                program(
                    choices=[
                        {"id": "bad", "outcomes": [{"p": "1/2", "facts": [["a"]]}]}
                    ]
                )
            )
        with self.assertRaisesRegex(ValueError, "canonical candidate"):
            Engine(
                program(),
                candidates=[
                    candidate("a"),
                    Candidate("alias", ("a",), ("not_alias",), "alias", "not alias"),
                ],
            )


class CacheTests(unittest.TestCase):
    def test_cache_includes_world_order_definition_and_backend(self):
        backend = OracleBackend({"a": "supported", "b": "refuted"})
        a, b = map(candidate, "ab")
        with tempfile.TemporaryDirectory() as directory:
            cache = DecisionCache(directory)
            _, key, hit = cache.evaluate(backend, "world", (a, b))
            self.assertFalse(hit)
            _, same, hit = DecisionCache(directory).evaluate(backend, "world", (a, b))
            self.assertTrue(hit)
            self.assertEqual(key, same)
            _, other, _ = cache.evaluate(backend, "world", (b, a))
            self.assertNotEqual(key, other)
            _, other, _ = cache.evaluate(backend, "different", (a, b))
            self.assertNotEqual(key, other)
            changed = Candidate(
                "a",
                a.atom,
                a.negative_atom,
                "different definition",
                a.negative_proposition,
            )
            _, other, _ = cache.evaluate(backend, "world", (changed, b))
            self.assertNotEqual(key, other)
            _, other, _ = cache.evaluate(
                OracleBackend({"a": "refuted", "b": "refuted"}), "world", (a, b)
            )
            self.assertNotEqual(key, other)


if __name__ == "__main__":
    unittest.main()
