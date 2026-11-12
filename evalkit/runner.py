"""Run a judge over every (case, rule) pair and build a report."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from .dataset import Dataset
from .metrics import disagreements, overall_accuracy, percentile, rule_metrics
from .models import Prediction, RunReport
from .providers import Judge, estimate_cost


def run(dataset: Dataset, judge: Judge, concurrency: int = 4, run_id: str | None = None) -> RunReport:
    pairs = [(case, rule) for case in dataset.cases for rule in dataset.rules]
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        predictions: list[Prediction] = list(pool.map(lambda pr: judge.judge(*pr), pairs))
    return build_report(dataset, predictions, provider=judge.name, model=judge.model, run_id=run_id)


def build_report(
    dataset: Dataset,
    predictions: list[Prediction],
    provider: str,
    model: str,
    run_id: str | None = None,
) -> RunReport:
    now = datetime.now(timezone.utc)
    total_in = sum(p.usage.input_tokens for p in predictions)
    total_out = sum(p.usage.output_tokens for p in predictions)
    latencies = [p.latency_ms for p in predictions if p.latency_ms]
    return RunReport(
        run_id=run_id or now.strftime("%Y%m%dT%H%M%SZ"),
        provider=provider,
        model=model,
        dataset=dataset.name,
        created_at=now.isoformat(timespec="seconds"),
        n_calls=len(predictions),
        overall_accuracy=overall_accuracy(dataset, predictions),
        rules=[rule_metrics(dataset, predictions, r.id) for r in dataset.rules],
        disagreements=disagreements(dataset, predictions),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        estimated_cost_usd=estimate_cost(model, total_in, total_out) if provider != "replay" else None,
        p50_latency_ms=percentile(latencies, 0.5),
        p95_latency_ms=percentile(latencies, 0.95),
        predictions=sorted(predictions, key=lambda p: (p.case_id, p.rule_id)),
    )
