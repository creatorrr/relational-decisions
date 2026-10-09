"""Small, exact finite positive-Datalog oracle; standard library only.

Ground atoms are tuples of strings. Variables start with '?'. Negative evidence
uses explicit predicates such as not_ready, never negation-as-failure.
"""
from fractions import Fraction
from itertools import product


def is_var(term):
    return isinstance(term, str) and term.startswith("?")


def validate_program(program):
    for atom in program["facts"]:
        assert not any(is_var(x) for x in atom), atom
    for choice in program["choices"]:
        assert sum(Fraction(o["p"]) for o in choice["outcomes"]) == 1
        assert all(0 <= Fraction(o["p"]) <= 1 for o in choice["outcomes"])
        assert all(not any(is_var(x) for x in a)
                   for o in choice["outcomes"] for a in o["facts"])
    for rule in program["rules"]:
        body_vars = {t for a in rule["body"] for t in a[1:] if is_var(t)}
        head_vars = {t for t in rule["head"][1:] if is_var(t)}
        assert head_vars <= body_vars, rule
        assert rule["body"], "Use facts for empty-body rules"


def closure(facts, rules):
    known = {tuple(a) for a in facts}
    while True:
        index = {}
        for atom in known:
            index.setdefault((atom[0], len(atom)), []).append(atom)
        added = set()
        for rule in rules:
            substitutions = [{}]
            for pattern in rule["body"]:
                following = []
                for subst in substitutions:
                    for atom in index.get((pattern[0], len(pattern)), ()):
                        extended = dict(subst)
                        for term, value in zip(pattern[1:], atom[1:]):
                            if is_var(term):
                                if term in extended and extended[term] != value:
                                    break
                                extended[term] = value
                            elif term != value:
                                break
                        else:
                            following.append(extended)
                substitutions = following
                if not substitutions:
                    break
            for subst in substitutions:
                added.add(tuple(subst.get(t, t) for t in rule["head"]))
        added -= known
        if not added:
            return known
        known.update(added)


def solve(program, observed_facts):
    """Enumerate categorical choices; condition; compute exact rational marginals."""
    validate_program(program)
    totals = {q["id"]: Fraction(0) for q in program["queries"]}
    evidence_mass = Fraction(0)
    world_count = 0
    choices = [c["outcomes"] for c in program["choices"]]
    for selected in product(*choices):
        weight = Fraction(1)
        facts = list(program["facts"]) + list(observed_facts)
        for outcome in selected:
            weight *= Fraction(outcome["p"])
            facts.extend(outcome["facts"])
        entailed = closure(facts, program["rules"])
        world_count += 1
        if any((tuple(e["atom"]) in entailed) != e["value"]
               for e in program.get("evidence", [])):
            continue
        evidence_mass += weight
        for query in program["queries"]:
            if tuple(query["atom"]) in entailed:
                totals[query["id"]] += weight
    assert evidence_mass > 0, "Impossible conditioning evidence"
    return {
        "query_probabilities": {k: str(v / evidence_mass) for k, v in totals.items()},
        "evidence_probability": str(evidence_mass),
        "enumerated_worlds": world_count,
    }


def prolog_atom(atom):
    def term(t):
        if is_var(t):
            return "V_" + t[1:]
        return "'" + t.replace("'", "''") + "'"
    predicate = term(atom[0])
    return predicate if len(atom) == 1 else predicate + "(" + ",".join(map(term, atom[1:])) + ")"


def to_problog(program, observed_facts):
    """Export the same program for an independent ProbLog cross-check."""
    lines = [prolog_atom(a) + "." for a in program["facts"] + list(observed_facts)]
    definitions = {(a[0], len(a) - 1) for a in program["facts"] + list(observed_facts)}
    for choice in program["choices"]:
        heads = []
        for i, outcome in enumerate(choice["outcomes"]):
            switch = ["choice_" + choice["id"], str(i)]
            heads.append(f"{float(Fraction(outcome['p'])):.17g}::{prolog_atom(switch)}")
            for atom in outcome["facts"]:
                lines.append(f"{prolog_atom(atom)} :- {prolog_atom(switch)}.")
                definitions.add((atom[0], len(atom) - 1))
        lines.append("; ".join(heads) + ".")
    for rule in program["rules"]:
        definitions.add((rule["head"][0], len(rule["head"]) - 1))
        lines.append(prolog_atom(rule["head"]) + " :- " + ", ".join(map(prolog_atom, rule["body"])) + ".")
    used = { (a[0], len(a) - 1) for r in program["rules"] for a in r["body"] }
    used |= {(q["atom"][0], len(q["atom"]) - 1) for q in program["queries"]}
    used |= {(e["atom"][0], len(e["atom"]) - 1) for e in program.get("evidence", [])}
    for predicate, arity in sorted(used - definitions):
        lines.append(prolog_atom([predicate] + [f"?u{i}" for i in range(arity)]) + " :- fail.")
    for e in program.get("evidence", []):
        lines.append(f"evidence({prolog_atom(e['atom'])}, {str(e['value']).lower()}).")
    for q in program["queries"]:
        lines.append(f"query({prolog_atom(q['atom'])}).")
    return "\n".join(lines) + "\n"
