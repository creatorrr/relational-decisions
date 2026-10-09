import unittest

from relational_decisions import Candidate
from relational_decisions.decisions import best_label, normalize
from relational_decisions.gliner import GLiNERBackend, check_encoder_runtime
from relational_decisions.prompts import (
    DEFAULT_PROMPT,
    LABEL_DESCRIPTIONS,
    PROMPTS,
    build_tasks,
    decode_scores,
)


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            Candidate(
                "a",
                ("ready", "oak"),
                ("not_ready", "oak"),
                "Oak is ready",
                "Oak is not ready",
            ),
            Candidate(
                "b",
                ("ready", "elm"),
                ("not_ready", "elm"),
                "Elm is ready",
                "Elm is not ready",
            ),
        ]

    def test_original_prompt_is_preserved(self):
        task = build_tasks(self.candidates, DEFAULT_PROMPT)[0]
        self.assertEqual(task.name, "decision_0")
        self.assertEqual(task.labels, LABEL_DESCRIPTIONS)
        self.assertEqual(
            task.instruction,
            "Assess explicit reports. Positive assertion: Oak is ready. Negative assertion: Oak is not ready.",
        )

    def test_binary_four_evidence_states_and_candidate_alignment(self):
        tasks = build_tasks(self.candidates, "binary-reports-v2")
        for positive, negative, expected in (
            (1, 0, "supported"),
            (0, 1, "refuted"),
            (1, 1, "both"),
            (0, 0, "unknown"),
        ):
            scores = {}
            for task in tasks:
                p = positive if task.channel == "positive" else negative
                if task.candidate_id == "b":
                    p = 1 - p
                scores[task.name] = {"no": 1 - p, "yes": p}
            decoded = decode_scores(tasks, scores)
            self.assertEqual(best_label(decoded["a"]), expected)
            self.assertEqual(sum(decoded["a"].values()), 1)
            self.assertNotEqual(best_label(decoded["b"]), expected)
        scores = {t.name: {"yes": 0.75, "no": 0.25} for t in tasks}
        self.assertEqual(
            decode_scores(tasks, scores)["a"],
            {
                "supported": 0.1875,
                "refuted": 0.1875,
                "both": 0.5625,
                "unknown": 0.0625,
            },
        )

    def test_categories_map_by_label_not_response_order(self):
        for prompt in PROMPTS[:-1]:
            tasks = build_tasks(self.candidates, prompt)
            scores = {}
            for task in tasks:
                expected = "supported" if task.candidate_id == "a" else "unknown"
                scores[task.name] = {
                    label: float(task.output_labels[label] == expected)
                    for label in reversed(task.labels)
                }
            decoded = decode_scores(tasks, scores)
            self.assertEqual(best_label(normalize(decoded["a"])), "supported")
            self.assertEqual(best_label(normalize(decoded["b"])), "unknown")
            del scores[tasks[0].name][next(iter(tasks[0].labels))]
            with self.assertRaises(ValueError):
                decode_scores(tasks, scores)

    def test_unknown_prompt_fails(self):
        with self.assertRaises(ValueError):
            build_tasks(self.candidates, "unregistered")

    def test_modernbert_checkpoint_cannot_silently_use_old_rotary_settings(self):
        config = {
            "encoder_config": {
                "model_type": "modernbert",
                "rope_parameters": {"sliding_attention": {"rope_theta": 160000}},
            }
        }
        with self.assertRaises(ValueError):
            check_encoder_runtime(config, "4.57.6")
        check_encoder_runtime(config, "5.17.0")
        check_encoder_runtime(
            {"encoder_config": {"model_type": "deberta-v2"}}, "4.57.6"
        )

    def test_prompt_switching_changes_cache_identity_and_restores_baseline(self):
        # This exercises metadata without loading optional ML dependencies.
        backend = GLiNERBackend.__new__(GLiNERBackend)
        backend._identity = {"backend": "test", "prompt_version": DEFAULT_PROMPT}
        baseline = dict(backend.identity)
        for prompt in PROMPTS[1:]:
            backend.set_prompt(prompt)
            self.assertEqual(backend.identity["prompt_version"], prompt)
            self.assertTrue(backend.identity["task_templates"])
            self.assertNotEqual(backend.identity, baseline)
        backend.set_prompt(DEFAULT_PROMPT)
        self.assertEqual(backend.identity, baseline)


if __name__ == "__main__":
    unittest.main()
