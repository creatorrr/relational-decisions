import unittest
from types import SimpleNamespace

from relational_decisions import Candidate
from relational_decisions.opendecision import (
    check_input_lengths,
    decode_responses,
    prepare_requests,
)
from relational_decisions.prompts import WORLD_PREFIX


class OpenDecisionTests(unittest.TestCase):
    def test_complete_questions_and_distinct_entities_survive_serialization(self):
        candidates = [
            Candidate(x, ("p", x), ("n", x), f"{x} is ready", f"{x} is not ready")
            for x in ("Oak", "Elm")
        ]
        tasks, texts = prepare_requests("No reports.", candidates)
        for task, text in zip(tasks, texts):
            self.assertIn(task.instruction, text)
            self.assertTrue(text.endswith(WORLD_PREFIX + "No reports."))
        # Simulate an API returning no before yes: map by name, preserve row ownership.
        policies = [
            {"no": 0, "yes": 1},
            {"no": 1, "yes": 0},
            {"no": 1, "yes": 0},
            {"no": 0, "yes": 1},
        ]
        result = decode_responses(tasks, policies)
        self.assertEqual(result["Oak"]["supported"], 1)
        self.assertEqual(result["Elm"]["refuted"], 1)
        with self.assertRaises(ValueError):
            decode_responses(tasks, policies[:-1])
        with self.assertRaises(ValueError):
            decode_responses(tasks, policies + policies[:1])

    def test_all_native_truncation_paths_fail_closed(self):
        encoder = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text, **kw: list(text))
        )
        cfg = SimpleNamespace(
            max_header_tokens=6,
            max_candidate_tokens=16,
            cross_candidate_tokens=16,
            max_state_tokens=12,
        )
        # answer = 6 tokens; 2 delimiters; "yes :: y" = 8; state = 12.
        self.assertEqual(
            check_input_lengths(encoder, cfg, 31, ["x" * 10], {"yes": "y"}), (12, 16)
        )
        with self.assertRaises(ValueError):
            check_input_lengths(encoder, cfg, 30, ["x" * 10], {"yes": "y"})
        for key, value in (
            ("max_header_tokens", 5),
            ("max_candidate_tokens", 15),
            ("cross_candidate_tokens", 15),
            ("max_state_tokens", 11),
        ):
            limits = SimpleNamespace(**{**vars(cfg), key: value})
            with self.assertRaises(ValueError):
                check_input_lengths(encoder, limits, 31, ["x" * 10], {"yes": "y"})


if __name__ == "__main__":
    unittest.main()
