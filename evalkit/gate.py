"""Regression gate: compare a run against a baseline and decide whether it may ship."""

from __future__ import annotations

from dataclasses import dataclass, field

from .models import RunReport


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def check(
    baseline: RunReport,
    current: RunReport,
    max_accuracy_drop: float = 0.02,
    max_recall_drop: float = 0.05,
    max_error_rate: float = 0.0,
) -> GateResult:
    """Fail when overall accuracy or any rule's fail-recall drops more than allowed,
    or when too many calls errored. Recall gets its own limit because a judge that
    stops catching failures is the regression an audit can't afford."""
    result = GateResult(passed=True)

    if baseline.dataset != current.dataset:
        result.failures.append(f"dataset mismatch: baseline {baseline.dataset!r}, current {current.dataset!r}")

    drop = baseline.overall_accuracy - current.overall_accuracy
    line = f"overall accuracy {baseline.overall_accuracy:.2%} -> {current.overall_accuracy:.2%}"
    (result.failures if drop > max_accuracy_drop + 1e-9 else result.notes).append(
        line + (f" (dropped {drop:.2%}, limit {max_accuracy_drop:.2%})" if drop > 0 else "")
    )

    base_rules = {r.rule_id: r for r in baseline.rules}
    for rule in current.rules:
        before = base_rules.get(rule.rule_id)
        if before is None:
            result.notes.append(f"{rule.rule_id}: new rule, no baseline")
            continue
        if before.fail_recall is None or rule.fail_recall is None:
            continue
        rdrop = before.fail_recall - rule.fail_recall
        msg = f"{rule.rule_id} fail-recall {before.fail_recall:.2%} -> {rule.fail_recall:.2%}"
        (result.failures if rdrop > max_recall_drop + 1e-9 else result.notes).append(msg)

    errors = sum(r.errors for r in current.rules)
    error_rate = errors / current.n_calls if current.n_calls else 0.0
    if error_rate > max_error_rate + 1e-9:
        result.failures.append(f"{errors} of {current.n_calls} calls errored ({error_rate:.2%}, limit {max_error_rate:.2%})")

    result.passed = not result.failures
    return result
