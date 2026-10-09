"""Canonical ordered multi-valued decision diagrams with exact weighted counts.

One variable represents one categorical choice; node children are its outcomes.
This directly encodes exclusivity and preserves shared choices across proofs.
"""

from fractions import Fraction
from functools import cache


class ResourceLimit(RuntimeError):
    """The configured budget was exhausted; no partial probability is returned."""


class Diagram:
    FALSE, TRUE = 0, 1

    def __init__(self, max_nodes=100_000):
        self.max_nodes = max_nodes
        self.choices = []
        self.nodes = [None, None]
        self._unique = {}
        self._apply_cache = {}
        self._not_cache = {0: 1, 1: 0}

    def register(self, name, outcomes, weights=None):
        if any(c["name"] == name for c in self.choices):
            raise ValueError(f"Duplicate choice: {name}")
        if not outcomes or len(set(outcomes)) != len(outcomes):
            raise ValueError("A choice needs distinct outcomes")
        if len(self.choices) >= 256:
            raise ResourceLimit(
                "This implementation supports at most 256 categorical choices"
            )
        variable = len(self.choices)
        self.choices.append(
            {"name": name, "outcomes": tuple(outcomes), "weights": None}
        )
        if weights is not None:
            self.set_weights(variable, weights)
        return variable

    def set_weights(self, variable, weights):
        values = tuple(Fraction(str(x)) for x in weights)
        choice = self.choices[variable]
        if (
            len(values) != len(choice["outcomes"])
            or any(p < 0 for p in values)
            or sum(values) != 1
        ):
            raise ValueError(
                "Choice weights must be nonnegative and sum exactly to one"
            )
        if choice["weights"] is not None and choice["weights"] != values:
            raise ValueError("A choice's weights are immutable once assessed")
        choice["weights"] = values

    def node(self, variable, children):
        children = tuple(children)
        if len(children) != len(self.choices[variable]["outcomes"]):
            raise ValueError("Wrong number of choice outcomes")
        if len(set(children)) == 1:
            return children[0]
        key = (variable, children)
        if key not in self._unique:
            if len(self.nodes) >= self.max_nodes:
                raise ResourceLimit("Decision diagram node budget exhausted")
            self._unique[key] = len(self.nodes)
            self.nodes.append(key)
        return self._unique[key]

    def event(self, variable, allowed):
        allowed = set(allowed)
        if not allowed <= set(range(len(self.choices[variable]["outcomes"]))):
            raise ValueError("Unknown outcome index")
        return self.node(
            variable,
            [int(i in allowed) for i in range(len(self.choices[variable]["outcomes"]))],
        )

    def and_(self, a, b):
        return self._apply("and", a, b)

    def or_(self, a, b):
        return self._apply("or", a, b)

    def _apply(self, op, a, b):
        if a > b:
            a, b = b, a
        if a == b:
            return a
        if op == "and":
            if a == 0:
                return 0
            if a == 1:
                return b
        else:
            if a == 0:
                return b
            if a == 1:
                return 1
        key = (op, a, b)
        if key in self._apply_cache:
            return self._apply_cache[key]
        av, ac = self.nodes[a]
        bv, bc = self.nodes[b]
        variable = min(av, bv)
        count = len(self.choices[variable]["outcomes"])
        children = [
            self._apply(
                op, ac[i] if av == variable else a, bc[i] if bv == variable else b
            )
            for i in range(count)
        ]
        result = self.node(variable, children)
        self._apply_cache[key] = result
        return result

    def not_(self, node):
        if node not in self._not_cache:
            variable, children = self.nodes[node]
            self._not_cache[node] = self.node(
                variable, [self.not_(c) for c in children]
            )
        return self._not_cache[node]

    def counter(self):
        """Create a memoized counter only after the requested assessments settle."""

        @cache
        def count(node):
            if node < 2:
                return Fraction(node)
            variable, children = self.nodes[node]
            weights = self.choices[variable]["weights"]
            if weights is None:
                raise ValueError(
                    "Attempted counting before a required assessment completed"
                )
            return sum(
                (p * count(c) for p, c in zip(weights, children) if p), Fraction()
            )

        return count

    def export(self, roots):
        """Serialize the reachable explanation DAG, rather than enumerating proofs."""
        reachable = set()
        todo = list(roots.values())
        while todo:
            node = todo.pop()
            if node < 2 or node in reachable:
                continue
            reachable.add(node)
            todo.extend(self.nodes[node][1])
        used = sorted({self.nodes[n][0] for n in reachable})
        return {
            "roots": roots,
            "terminals": {"0": False, "1": True},
            "nodes": {
                str(n): {"choice": self.nodes[n][0], "children": self.nodes[n][1]}
                for n in sorted(reachable)
            },
            "choices": {
                str(v): {
                    **self.choices[v],
                    "weights": None
                    if self.choices[v]["weights"] is None
                    else list(map(str, self.choices[v]["weights"])),
                }
                for v in used
            },
        }
