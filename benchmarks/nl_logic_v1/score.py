"""Evaluate open train/dev partitions. Heldout is deliberately not opened here."""
import argparse
import collections
from fractions import Fraction
import json
import math
from pathlib import Path

from logic import solve

ROOT = Path(__file__).resolve().parent
LABELS = ("supported", "refuted", "both", "unknown")


def read_jsonl(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert len({r["id"] for r in rows}) == len(rows), f"Duplicate IDs: {path}"
    return {r["id"]:r for r in rows}


def summarize(records):
    confusion = {x:{y:0 for y in LABELS} for x in LABELS}
    errors = []
    for r in records:
        for truth,pred in r["labels"]:
            confusion[truth][pred] += 1
        errors.extend(r["probability_errors"])
    count = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[x][x] for x in LABELS)
    f1s = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[x][label] for x in LABELS if x != label)
        fn = sum(confusion[label][x] for x in LABELS if x != label)
        denom = 2*tp+fp+fn
        f1s.append(2*tp/denom if denom else 0.0)
    return {
        "programs":len(records), "assessments":count, "queries":len(errors),
        "assessment_accuracy":correct/count if count else None,
        "assessment_macro_f1":sum(f1s)/len(f1s),
        "query_probability_mae":sum(errors)/len(errors) if errors else None,
        "query_probability_max_error":max(errors,default=None),
        "query_probability_within_1e-9":sum(e <= 1e-9 for e in errors)/len(errors) if errors else None,
        "confusion_matrix":confusion,
    }


def evaluate(inputs, gold, predictions, derive_probabilities=False):
    if set(inputs) != set(gold) or set(inputs) != set(predictions):
        raise ValueError("Predictions must cover every input ID exactly once; missing/extra IDs are rejected.")
    records = []
    normalized_predictions = {}
    for pid,inp in inputs.items():
        target,pred = gold[pid],predictions[pid]
        labels = pred["assessments"]
        if set(labels) != set(target["assessments"]) or any(v not in LABELS for v in labels.values()):
            raise ValueError(f"Invalid/missing assessment labels for {pid}")
        if derive_probabilities:
            facts = []
            for candidate in inp["candidates"]:
                state = labels[candidate["id"]]
                if state in ("supported","both"):
                    facts.append(candidate["atom"])
                if state in ("refuted","both"):
                    facts.append(candidate["negative_atom"])
            probabilities = solve(inp["program"],facts)["query_probabilities"]
        else:
            probabilities = pred["query_probabilities"]
        if set(probabilities) != set(target["query_probabilities"]):
            raise ValueError(f"Invalid/missing query IDs for {pid}")
        probabilities = {q:float(Fraction(str(p))) for q,p in probabilities.items()}
        if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
            raise ValueError(f"Probabilities outside [0,1] for {pid}")
        records.append({
            "id":pid, "family":inp["family"], "domain":inp["domain"],
            "labels":[(truth,labels[q]) for q,truth in target["assessments"].items()],
            "probability_errors":[abs(probabilities[q]-float(Fraction(p))) for q,p in target["query_probabilities"].items()],
        })
        normalized_predictions[pid] = {"assessments":labels,"probabilities":probabilities}
    grouped = collections.defaultdict(list)
    for pid,inp in inputs.items():
        grouped[inp["group_id"]].append(normalized_predictions[pid])
    label_agreement = []
    probability_differences = []
    for group in grouped.values():
        assert len(group) == 2
        a,b = group
        label_agreement.extend(a["assessments"][q] == b["assessments"][q] for q in a["assessments"])
        probability_differences.extend(abs(a["probabilities"][q]-b["probabilities"][q]) for q in a["probabilities"])
    return {
        "overall":summarize(records),
        "by_family":{f:summarize([r for r in records if r["family"]==f]) for f in sorted({r["family"] for r in records})},
        "by_domain":{d:summarize([r for r in records if r["domain"]==d]) for d in sorted({r["domain"] for r in records})},
        "paraphrase_assessment_agreement":sum(label_agreement)/len(label_agreement),
        "paraphrase_query_probability_mean_difference":sum(probability_differences)/len(probability_differences),
        "probability_source":"oracle_on_predicted_assessments" if derive_probabilities else "submitted_engine_outputs",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",choices=["train","dev"],required=True)
    parser.add_argument("--predictions",type=Path,required=True)
    parser.add_argument("--derive-probabilities",action="store_true")
    args = parser.parse_args()
    result = evaluate(
        read_jsonl(ROOT / "data" / f"{args.split}.inputs.jsonl"),
        read_jsonl(ROOT / "data" / f"{args.split}.gold.jsonl"),
        read_jsonl(args.predictions), args.derive_probabilities,
    )
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
