"""Native-interface sanity checks on synthetic statements, without bench gold."""
import argparse
import json
from pathlib import Path

from adapters import K2Backend, LuxBackend
from relational_decisions import Candidate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["k2", "lux-q4"], required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--server-binary")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    backend = K2Backend(args.model_path) if args.model == "k2" else LuxBackend(args.model_path, args.server_binary)
    try:
        candidates = [Candidate("ada", ("p",), ("n",), "Ada is a member.", "Ada is not a member."),
                      Candidate("bea", ("q",), ("m",), "Bea is a member.", "Bea is not a member.")]
        world = "Ada is a member. Bea is not a member."
        grouped = backend.assess(world, candidates)
        repeated = backend.assess(world, candidates)
        individual = {c.id: backend.assess(world, [c])[c.id] for c in candidates}
        difference = max(abs(grouped[c.id][k] - individual[c.id][k]) for c in candidates for k in grouped[c.id])
        args.output.write_text(json.dumps({"model": backend.identity, "world": world,
            "grouped": grouped, "repeated": repeated, "individual": individual,
            "repeat_bitwise_equal": grouped == repeated,
            "group_vs_single_max_absolute_error": difference,
            "scope": "synthetic interface smoke; not accuracy evidence",
            "resources": backend.resources()}, indent=2) + "\n")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
