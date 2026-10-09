import unittest
from types import SimpleNamespace

from relational_decisions import Candidate
from relational_decisions.openjev import check_state_length, option_texts
from relational_decisions.prompts import build_tasks


class OpenJevTests(unittest.TestCase):
    def test_native_options_preserve_frozen_labels_order_and_descriptions(self):
        c = Candidate("a", ("p",), ("n",), "Oak is ready", "Oak is not ready")
        task = build_tasks([c], "binary-reports-v2")[0]
        self.assertEqual(
            option_texts(task),
            [
                "yes :: The reports explicitly state this statement.",
                "no :: The reports do not state this statement.",
            ],
        )

    def test_native_state_truncation_is_rejected_at_its_boundary(self):
        tokenizer = SimpleNamespace(encode=lambda text, **kw: list(text))
        self.assertEqual(check_state_length(tokenizer, "abc", 3), 3)
        with self.assertRaises(ValueError):
            check_state_length(tokenizer, "abcd", 3)


if __name__ == "__main__":
    unittest.main()
