"""Score predictions against the golden labels."""

from __future__ import annotations

from .dataset import Dataset
from .models import Disagreement, Prediction, RuleMetrics


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def rule_metrics(dataset: Dataset, predictions: list[Prediction], rule_id: str) -> RuleMetrics:
    """'fail' is the positive class: a missed failure is the costly mistake in an audit."""
    tp = fp = fn = tn = errors = correct = n = 0
    for p in predictions:
        if p.rule_id != rule_id:
            continue
        n += 1
        expected = dataset.label_for(p.case_id, p.rule_id).label
        if p.verdict is None:
            errors += 1
            # A failed call counts as wrong. It can't be counted as a catch either way.
            if expected == "fail":
                fn += 1
            continue
        if p.verdict == expected:
            correct += 1
        if p.verdict == "fail" and expected == "fail":
            tp += 1
        elif p.verdict == "fail" and expected == "pass":
            fp += 1
        elif p.verdict == "pass" and expected == "fail":
            fn += 1
        else:
            tn += 1
    return RuleMetrics(
        rule_id=rule_id,
        n=n,
        accuracy=_ratio(correct, n) or 0.0,
        fail_precision=_ratio(tp, tp + fp),
        fail_recall=_ratio(tp, tp + fn),
        confusion={"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        errors=errors,
    )


def disagreements(dataset: Dataset, predictions: list[Prediction]) -> list[Disagreement]:
    out = []
    for p in predictions:
        expected = dataset.label_for(p.case_id, p.rule_id).label
        if p.verdict != expected:
            out.append(
                Disagreement(
                    case_id=p.case_id,
                    rule_id=p.rule_id,
                    expected=expected,
                    got=p.verdict,
                    reason=p.error or p.reason,
                )
            )
    return sorted(out, key=lambda d: (d.rule_id, d.case_id))


def overall_accuracy(dataset: Dataset, predictions: list[Prediction]) -> float:
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if p.verdict == dataset.label_for(p.case_id, p.rule_id).label)
    return round(correct / len(predictions), 4)


def percentile(values: list[int], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]
