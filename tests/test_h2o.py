import unittest
from types import SimpleNamespace

from relational_decisions.h2o import label_token_ids


class H2OReadoutTests(unittest.TestCase):
    def tokenizer(self, rows):
        return SimpleNamespace(encode=lambda text, **kwargs: rows[text])

    def test_native_answer_slot_preserves_label_order(self):
        tok = self.tokenizer(
            {"Answer:": [1, 2], "Answer: A": [1, 2, 9], "Answer: B": [1, 2, 7]}
        )
        self.assertEqual(label_token_ids(tok, "Answer:", ["A", "B"]), ([1, 2], [9, 7]))

    def test_retokenized_prefix_is_rejected(self):
        tok = self.tokenizer({"Answer:": [1, 2], "Answer: A": [1, 3, 9]})
        with self.assertRaises(ValueError):
            label_token_ids(tok, "Answer:", ["A"])

    def test_multi_token_label_is_rejected(self):
        tok = self.tokenizer({"Answer:": [1, 2], "Answer: A": [1, 2, 8, 9]})
        with self.assertRaises(ValueError):
            label_token_ids(tok, "Answer:", ["A"])

    def test_colliding_label_ids_are_rejected(self):
        tok = self.tokenizer(
            {"Answer:": [1, 2], "Answer: A": [1, 2, 9], "Answer: B": [1, 2, 9]}
        )
        with self.assertRaises(ValueError):
            label_token_ids(tok, "Answer:", ["A", "B"])


if __name__ == "__main__":
    unittest.main()
