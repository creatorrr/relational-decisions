"""Versioned language-grounding formulations, independent of ML packages."""

from dataclasses import dataclass

from .decisions import LABELS

DEFAULT_PROMPT = "explicit-reports-v1"
PROMPTS = (
    DEFAULT_PROMPT,
    "direct-status-v2",
    "concrete-options-v2",
    "binary-reports-v2",
)
WORLD_PREFIX = "Unordered reports about one snapshot. No report has priority. Missing information is unknown.\n\n"
LABEL_DESCRIPTIONS = {
    "supported": "Only the positive assertion is explicitly reported.",
    "refuted": "Only the negative assertion is explicitly reported.",
    "both": "Both assertions are explicitly reported, so reports conflict.",
    "unknown": "Neither assertion is explicitly reported; information is missing.",
}
STATUS_LABELS = {
    "affirmed": "supported",
    "denied": "refuted",
    "conflicting": "both",
    "not_mentioned": "unknown",
}


@dataclass(frozen=True)
class Task:
    name: str
    candidate_id: str
    channel: str
    instruction: str
    labels: dict[str, str]
    output_labels: dict[str, str]


def build_tasks(candidates, prompt=DEFAULT_PROMPT):
    if prompt not in PROMPTS:
        raise ValueError(f"Unknown grounding prompt: {prompt}")
    tasks = []
    for i, c in enumerate(candidates):
        if prompt == "binary-reports-v2":
            for channel, statement in (
                ("positive", c.proposition),
                ("negative", c.negative_proposition),
            ):
                tasks.append(
                    Task(
                        f"decision_{i}_{channel}",
                        c.id,
                        channel,
                        f'Is "{statement}" explicitly stated in the reports? '
                        "Match the exact named entity. A paraphrase counts as stated.",
                        {
                            "yes": "The reports explicitly state this statement.",
                            "no": "The reports do not state this statement.",
                        },
                        {},
                    )
                )
            continue
        if prompt == DEFAULT_PROMPT:
            instruction = (
                f"Assess explicit reports. Positive assertion: {c.proposition}. "
                f"Negative assertion: {c.negative_proposition}."
            )
            labels = dict(LABEL_DESCRIPTIONS)
            mapping = {label: label for label in LABELS}
        elif prompt == "direct-status-v2":
            instruction = (
                f'What do the reports say about this claim: "{c.proposition}"? '
                "Check only the exact named entity."
            )
            labels = {
                "affirmed": "The claim is stated and is not denied.",
                "denied": "The claim is denied and is not stated.",
                "conflicting": "The claim is both stated and denied in different reports.",
                "not_mentioned": "The reports neither state nor deny the claim.",
            }
            mapping = dict(STATUS_LABELS)
        else:
            instruction = "Which statements are explicitly reported? Check the exact named entity."
            p, n = c.proposition, c.negative_proposition
            labels = {
                "affirmed": f'The reports state "{p}" but do not state "{n}".',
                "denied": f'The reports state "{n}" but do not state "{p}".',
                "conflicting": f'The reports state both "{p}" and "{n}".',
                "not_mentioned": f'The reports state neither "{p}" nor "{n}".',
            }
            mapping = dict(STATUS_LABELS)
        tasks.append(
            Task(f"decision_{i}", c.id, "category", instruction, labels, mapping)
        )
    return tasks


def decode_scores(tasks, scores):
    """Map model labels to the engine contract; binary heads use a product joint.

    The product assumes independence between positive-report and negative-report
    judgments. It is useful for hard decisions but is not a calibrated joint
    distribution. The engine's existing independence assumption between distinct
    candidates is an additional assumption when using soft inference.
    """
    result, binary = {}, {}
    for task in tasks:
        values = scores[task.name]
        if set(values) != set(task.labels):
            raise ValueError(f"Missing or extra model labels for {task.name}")
        if task.channel == "category":
            result[task.candidate_id] = {
                target: values[source] for source, target in task.output_labels.items()
            }
        else:
            binary.setdefault(task.candidate_id, {})[task.channel] = values
    for cid, channels in binary.items():
        p, n = channels["positive"], channels["negative"]
        result[cid] = {
            "supported": p["yes"] * n["no"],
            "refuted": p["no"] * n["yes"],
            "both": p["yes"] * n["yes"],
            "unknown": p["no"] * n["no"],
        }
    return result
