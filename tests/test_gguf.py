"""Guard against invalid/truncated label readouts without downloading models."""

import math
import unittest

from relational_decisions.gguf import label_probabilities


class GGUFReadoutTests(unittest.TestCase):
    def response(self, scores):
        return {"truncated": False, "completion_probabilities": [{
            "top_logprobs": [{"id": i, "logprob": p} for i, p in scores.items()]
        }]}

    def test_full_vocabulary_normalizer_cancels(self):
        logits = {357: 2.0, 417: -1.0, 999: 5.0}
        normalizer = math.log(sum(math.exp(v) for v in logits.values()))
        response = self.response({i: v - normalizer for i, v in logits.items()})
        actual = label_probabilities(response, [357, 417], 0.8)
        self.assertAlmostEqual(actual[0], 1 / (1 + math.exp(-3 / 0.8)))
        self.assertAlmostEqual(sum(actual), 1.0)

    def test_missing_answer_and_truncation_fail_closed(self):
        for response in (
            self.response({357: -1}),
            self.response({357: float("nan"), 417: -2}),
            {**self.response({357: -1, 417: -2}), "truncated": True},
        ):
            with self.assertRaises(ValueError):
                label_probabilities(response, [357, 417], 0.8)
