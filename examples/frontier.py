"""Run a tiny program and inspect when its neural goals become ready."""

import argparse
import json

from relational_decisions import Engine, EngineConfig, OracleBackend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gliner",
        action="store_true",
        help="Use the local model instead of the exact demonstration labels",
    )
    args = parser.parse_args()
    candidates = [
        {
            "id": "move",
            "atom": ["requests_move", "room1408"],
            "negative_atom": ["not_requests_move", "room1408"],
            "proposition": "The guest in room 1408 requests a room change",
            "negative_proposition": "The guest in room 1408 does not request a room change",
        },
        {
            "id": "repair",
            "atom": ["needs_repair", "room1408"],
            "negative_atom": ["not_needs_repair", "room1408"],
            "proposition": "The guest reports that the AC in room 1408 needs repair",
            "negative_proposition": "The guest reports that the AC in room 1408 does not need repair",
        },
        {
            "id": "vacancy",
            "atom": ["vacancy", "suite208"],
            "negative_atom": ["not_vacancy", "suite208"],
            "proposition": "Suite 208 is available",
            "negative_proposition": "Suite 208 is unavailable",
        },
    ]
    program = {
        "facts": [],
        "rules": [
            {
                "head": ["helped", "?room"],
                "body": [
                    ["requests_move", "?room"],
                    ["vacancy", "suite208"],
                    ["move_approved"],
                ],
            },
            {
                "head": ["helped", "?room"],
                "body": [["needs_repair", "?room"], ["technician_available"]],
            },
        ],
        "choices": [
            {
                "id": "approval",
                "outcomes": [
                    {"p": "4/5", "facts": [["move_approved"]]},
                    {"p": "1/5", "facts": []},
                ],
            },
            {
                "id": "technician",
                "outcomes": [
                    {"p": "3/5", "facts": [["technician_available"]]},
                    {"p": "2/5", "facts": []},
                ],
            },
        ],
        "queries": [{"id": "guest_helped", "atom": ["helped", "room1408"]}],
        "evidence": [],
    }
    world = (
        "The guest in room 1408 requests a room change. "
        "The guest reports that the AC in room 1408 needs repair. Suite 208 is available."
    )
    if args.gliner:
        from relational_decisions.gliner import GLiNERBackend

        backend = GLiNERBackend()
    else:
        backend = OracleBackend({c["id"]: "supported" for c in candidates})
    result = Engine(
        program,
        candidates=candidates,
        world=world,
        backend=backend,
        config=EngineConfig(batch_size=4),
    ).run()
    print(
        json.dumps(
            {
                "backend": "gliner" if args.gliner else "oracle",
                "probability": str(result.query_probabilities["guest_helped"]),
                "batches": [t["candidate_ids"] for t in result.trace],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
