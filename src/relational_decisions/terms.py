"""Flat, function-free first-order terms and variant keys."""

from dataclasses import dataclass
from itertools import count


@dataclass(frozen=True)
class Var:
    index: int


class Terms:
    def __init__(self):
        self._ids = count()

    def parse(self, atom, variables=None):
        variables = {} if variables is None else variables
        if not isinstance(atom, (list, tuple)) or not atom:
            raise ValueError("An atom must be a nonempty sequence of strings")
        if any(not isinstance(t, str) or not t for t in atom) or atom[0].startswith(
            "?"
        ):
            raise ValueError(
                "Predicates and constants must be nonempty strings; only arguments can be variables"
            )
        result = [atom[0]]
        for term in atom[1:]:
            if term.startswith("?"):
                if term not in variables:
                    variables[term] = Var(next(self._ids))
                result.append(variables[term])
            else:
                result.append(term)
        return tuple(result)


def walk(term, subst):
    while isinstance(term, Var) and term in subst:
        term = subst[term]
    return term


def reify(atom, subst):
    return tuple(walk(term, subst) for term in atom)


def unify(left, right, subst):
    if len(left) != len(right) or left[0] != right[0]:
        return None
    extended = dict(subst)
    for a, b in zip(left[1:], right[1:]):
        a, b = walk(a, extended), walk(b, extended)
        if a == b:
            continue
        if isinstance(a, Var):
            extended[a] = b
        elif isinstance(b, Var):
            extended[b] = a
        else:
            return None
    return extended


def variant(atoms):
    """Normalize variable names jointly, preserving repeated-variable identity."""
    names = {}
    result = []
    for atom in atoms:
        row = []
        for term in atom:
            if isinstance(term, Var):
                if term not in names:
                    names[term] = len(names)
                row.append(("var", names[term]))
            else:
                row.append(("constant", term))
        result.append(tuple(row))
    return tuple(result)


def ground(atom):
    return not any(isinstance(t, Var) for t in atom)
