"""Independently check matched quick8 predictions using only public dev data."""

import argparse
import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LABELS = ('supported', 'refuted', 'both', 'unknown')


def rows(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len({row['id'] for row in records}) == len(records), path
    return {row['id']: row for row in records}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


REFERENCE = module('independent_logic', ROOT / 'benchmarks/nl_logic_v1/logic.py')
INPUTS = dict(list(rows(ROOT / 'benchmarks/nl_logic_v1/data/dev.inputs.jsonl').items())[:8])
GOLD = rows(ROOT / 'benchmarks/nl_logic_v1/data/dev.gold.jsonl')
REQUESTS = rows(ROOT / 'experiments/prompt_v2/decide-340m-dev.traces.jsonl')


def verify(prefix):
    def artifact(name):
        return prefix / name if prefix.is_dir() else Path(str(prefix) + '.' + name)

    predictions = rows(artifact('predictions.jsonl'))
    assert set(predictions) == set(INPUTS), 'Prediction IDs differ from frozen quick8'
    confusion = {truth: {pred: 0 for pred in LABELS} for truth in LABELS}
    errors, exact = [], 0
    for pid, inp in INPUTS.items():
        pred, truth = predictions[pid], GOLD[pid]
        assert set(pred['assessments']) == set(truth['assessments']), pid
        assert set(pred['query_probabilities']) == set(truth['query_probabilities']), pid
        facts = []
        for candidate in inp['candidates']:
            label = pred['assessments'][candidate['id']]
            assert label in LABELS
            confusion[truth['assessments'][candidate['id']]][label] += 1
            if label in ('supported', 'both'):
                facts.append(candidate['atom'])
            if label in ('refuted', 'both'):
                facts.append(candidate['negative_atom'])
        reference = REFERENCE.solve(inp['program'], facts)['query_probabilities']
        for qid, value in reference.items():
            submitted = Fraction(pred['query_probabilities'][qid])
            assert submitted == Fraction(value), (pid, qid, submitted, value)
            target = Fraction(truth['query_probabilities'][qid])
            exact += submitted == target
            errors.append(abs(submitted - target))
    total = sum(sum(row.values()) for row in confusion.values())
    correct = sum(confusion[label][label] for label in LABELS)
    f1 = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[x][label] for x in LABELS if x != label)
        fn = sum(confusion[label][x] for x in LABELS if x != label)
        f1.append(Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0))
    metrics = {
        'programs': len(predictions),
        'assessments': total,
        'queries': len(errors),
        'assessment_accuracy': correct / total,
        'assessment_macro_f1': float(sum(f1) / len(LABELS)),
        'query_probability_mae': float(sum(errors) / len(errors)),
        'query_probability_max_error': float(max(errors)),
        'query_probability_within_1e-9': sum(float(e) <= 1e-9 for e in errors) / len(errors),
        'confusion_matrix': confusion,
    }
    recorded = json.loads(artifact('metrics.json').read_text())['overall']
    for key, expected in metrics.items():
        if isinstance(expected, float):
            assert abs(recorded[key] - expected) < 1e-12, (key, recorded[key], expected)
        else:
            assert recorded[key] == expected, key
    trace_path = artifact('traces.jsonl')
    if trace_path.exists():
        traces = rows(trace_path)
        assert set(traces) == set(INPUTS)
        for pid, trace in traces.items():
            assert trace['assessments'] == predictions[pid]['assessments']
            assert trace['query_probabilities'] == predictions[pid]['query_probabilities']
            assert set(trace['distributions']) == set(predictions[pid]['assessments'])
            for cid, distribution in trace['distributions'].items():
                assert set(distribution) == set(LABELS)
                weights = {label: Fraction(value) for label, value in distribution.items()}
                assert sum(weights.values()) == 1 and min(weights.values()) >= 0
                assert max(LABELS, key=lambda label: weights[label]) == predictions[pid]['assessments'][cid]
            if 'replayed_model_requests' in trace:
                replayed = trace['replayed_model_requests']
                assert [r['candidate_ids'] for r in replayed] == [
                    r['candidate_ids'] for r in REQUESTS[pid]['trace']
                ], (pid, 'Request groups differ')
                assert all(r['cache_hit'] is False for r in replayed)
    metadata = json.loads(artifact('metadata.json').read_text())
    assert metadata['status'] == 'complete'
    assert metadata.get('heldout_opened') is False
    if 'input_ids' in metadata:
        assert metadata['input_ids'] == list(INPUTS)
    source_hashes = {
        'input_sha256': ROOT / 'benchmarks/nl_logic_v1/data/dev.inputs.jsonl',
        'request_groups_sha256': ROOT / 'experiments/prompt_v2/decide-340m-dev.traces.jsonl',
        'driver_sha256': ROOT / 'experiments/additional_models/run_quick.py',
        'adapter_sha256': ROOT / 'experiments/additional_models/adapters.py',
    }
    if 'request_groups_sha256' in metadata:
        for key, path in source_hashes.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata[key], key
    return {
        'path': str(prefix),
        'correct_assessments': correct,
        'assessment_count': total,
        'exact_queries': exact,
        'query_count': len(errors),
        'independent_exact_queries_verified': len(errors),
        'all_recorded_metrics_match': True,
        'per_label_correct': {label: [confusion[label][label], sum(confusion[label].values())] for label in LABELS},
        'metrics': metrics,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('prefixes', nargs='+', type=Path)
    args = parser.parse_args()
    print(json.dumps([verify(prefix) for prefix in args.prefixes], indent=2))
