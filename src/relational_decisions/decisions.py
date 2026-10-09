"""Ground proposition registry, backend contract, and exact-input batch cache."""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Protocol

LABELS = ("supported", "refuted", "both", "unknown")


@dataclass(frozen=True)
class Candidate:
    id: str
    atom: tuple[str, ...]
    negative_atom: tuple[str, ...]
    proposition: str
    negative_proposition: str

    @classmethod
    def from_dict(cls, value):
        return cls(
            value["id"],
            tuple(value["atom"]),
            tuple(value["negative_atom"]),
            value["proposition"],
            value["negative_proposition"],
        )


class DecisionBackend(Protocol):
    @property
    def identity(self) -> dict: ...

    def assess(self, world: str, candidates: tuple[Candidate, ...]) -> dict: ...


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def normalize(distribution):
    if set(distribution) != set(LABELS):
        raise ValueError("A neural assessment must supply all four label weights")
    try:
        values = {label: Fraction(str(distribution[label])) for label in LABELS}
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Assessment weights must be finite numbers") from exc
    total = sum(values.values())
    if total <= 0 or any(p < 0 for p in values.values()):
        raise ValueError("Assessment weights must be nonnegative with positive total")
    return {k: v / total for k, v in values.items()}


def best_label(distribution):
    # Stable tie breaking follows the declared label order.
    return max(LABELS, key=lambda label: distribution[label])


class OracleBackend:
    """Testing adapter: supplies gold categories through the neural interface."""

    def __init__(self, labels):
        self.labels = dict(labels)
        if any(v not in LABELS for v in labels.values()):
            raise ValueError("Unknown oracle label")

    @property
    def identity(self):
        return {"backend": "oracle-v1", "labels_hash": fingerprint(self.labels)}

    def assess(self, world, candidates):
        return {
            c.id: {label: int(label == self.labels[c.id]) for label in LABELS}
            for c in candidates
        }


class DecisionCache:
    """Content-addressed cache; joint schema order is deliberately significant."""

    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else None
        self.memory = {}

    def evaluate(self, backend, world, candidates):
        payload = {
            "version": 1,
            "backend": backend.identity,
            "world": world,
            "ordered_candidates": [asdict(c) for c in candidates],
        }
        key = fingerprint(payload)
        path = self.directory / f"{key}.json" if self.directory is not None else None
        cached = key in self.memory
        if cached:
            raw = self.memory[key]
        elif path is not None and path.exists():
            record = json.loads(path.read_text())
            if record["key"] != key:
                raise ValueError("Decision cache key mismatch")
            raw = record["assessments"]
            cached = True
        else:
            raw = backend.assess(world, candidates)
        if set(raw) != {c.id for c in candidates}:
            raise ValueError("Backend returned missing or extra candidate IDs")
        normalized = {cid: normalize(scores) for cid, scores in raw.items()}
        encoded = {
            cid: {label: str(p) for label, p in scores.items()}
            for cid, scores in normalized.items()
        }
        self.memory[key] = encoded
        if path is not None and not cached:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", dir=path.parent, delete=False
            ) as file:
                temp = Path(file.name)
                json.dump({"key": key, "assessments": encoded}, file, sort_keys=True)
            os.replace(temp, path)
        return normalized, key, cached
