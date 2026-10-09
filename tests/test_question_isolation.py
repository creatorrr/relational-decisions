import unittest

from experiments.question_isolation.isolated import question_rows
from relational_decisions import Candidate
from relational_decisions.opendecision import prepare_requests
from relational_decisions.prompts import WORLD_PREFIX, build_tasks


class IsolationTests(unittest.TestCase):
    def test_native_isolation_preserves_original_task_identity_and_wording(self):
        candidates = tuple(
            Candidate(x, ("p", x), ("n", x), f"{x} is ready", f"{x} is not ready")
            for x in ("Oak", "Elm")
        )
        tasks = build_tasks(candidates, "binary-reports-v2")
        rows = question_rows("Reports.", candidates, "isolated-native")
        self.assertEqual(len(rows), 4)
        for expected, (task, name, instruction, text) in zip(tasks, rows):
            self.assertEqual(task, expected)
            self.assertEqual(name, expected.name)
            self.assertEqual(instruction, expected.instruction)
            self.assertEqual(text, WORLD_PREFIX + "Reports.")
        self.assertEqual(rows[2][1], "decision_1_positive")

    def test_state_layout_matches_opendecision_and_retains_result_mapping(self):
        candidates = (
            Candidate("oak", ("p",), ("n",), "Oak is ready", "Oak is not ready"),
        )
        tasks, texts = prepare_requests("Reports.", candidates)
        rows = question_rows("Reports.", candidates, "isolated-question-in-state")
        self.assertEqual([r[0] for r in rows], tasks)
        self.assertEqual([r[3] for r in rows], texts)
        self.assertTrue(all(r[1] == "answer" and r[2] is None for r in rows))
        with self.assertRaises(ValueError):
            question_rows("Reports.", candidates, "unknown")


if __name__ == "__main__":
    unittest.main()
