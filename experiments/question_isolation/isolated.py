"""GLiNER isolation controls; native scoring with one question in each row."""

from relational_decisions.gliner import GLiNERBackend
from relational_decisions.opendecision import STATE_TEMPLATE
from relational_decisions.prompts import WORLD_PREFIX, build_tasks, decode_scores

LAYOUTS = ("isolated-native", "isolated-question-in-state")


def question_rows(world, candidates, layout):
    if layout not in LAYOUTS:
        raise ValueError(f"Unknown isolation layout: {layout}")
    rows = []
    for task in build_tasks(candidates, "binary-reports-v2"):
        name, instruction, text = task.name, task.instruction, WORLD_PREFIX + world
        if layout == "isolated-question-in-state":
            name, instruction = "answer", None
            text = STATE_TEMPLATE.format(
                instruction=task.instruction, world_prefix=WORLD_PREFIX, world=world
            )
        rows.append((task, name, instruction, text))
    return rows


class IsolatedGLiNERBackend(GLiNERBackend):
    def __init__(self, model, revision, *, layout, threads=4, max_tokens=4096):
        if layout not in LAYOUTS:
            raise ValueError(f"Unknown isolation layout: {layout}")
        super().__init__(
            model,
            revision,
            threads=threads,
            max_tokens=max_tokens,
            prompt="binary-reports-v2",
        )
        self.layout = layout
        self._identity.update(
            {
                "isolation_version": "independent-question-rows-v1",
                "question_layout": layout,
                "question_batch_size": 8,
                "questions_per_attention_sequence": 1,
                "state_template": STATE_TEMPLATE
                if layout.endswith("question-in-state")
                else "{world_prefix}{world}",
                "task_name": "answer"
                if layout.endswith("question-in-state")
                else "original baseline slot name",
            }
        )

    def assess(self, world, candidates):
        from gliner2.classification import ClassificationConfig, ClassificationSchema
        from gliner2.classification.compiler import compile_schema

        rows = question_rows(world, candidates, self.layout)
        if not rows:
            return {}
        schemas = []
        for task, name, instruction, _ in rows:
            schema = ClassificationSchema()
            schema.single(
                name, task.labels, instruction=instruction, activation="softmax"
            )
            schemas.append(compile_schema(schema))
        texts = [row[3] for row in rows]
        # Validate the same batches the native scorer will encode.
        for offset in range(0, len(rows), 8):
            batch = self.model.processor.collate_fn_inference(
                [
                    (t, s.build())
                    for t, s in zip(
                        texts[offset : offset + 8], schemas[offset : offset + 8]
                    )
                ],
                architecture=self.model.architecture,
            )
            if any(int(mask.sum()) > self.max_tokens for mask in batch.attention_mask):
                raise ValueError("Isolated question exceeds the declared token limit")
            if any(len(tasks) != 1 for tasks in batch.schema_tokens_list):
                raise ValueError(
                    "Expected exactly one schema task in each attention row"
                )
        scores = self.classifier.batch_score(
            texts, schemas, config=ClassificationConfig(batch_size=8)
        )
        if len(scores) != len(rows):
            raise ValueError("Question/response count mismatch")
        return decode_scores(
            [row[0] for row in rows],
            {
                task.name: {
                    label: result.probability(name, label) for label in task.labels
                }
                for (task, name, _, _), result in zip(rows, scores)
            },
        )
