"""Command line: evalkit run | gate | show."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .dataset import load_dataset
from .gate import check
from .models import RunReport
from .providers import AnthropicJudge, ReplayJudge
from .runner import run


def _load_report(path: str) -> RunReport:
    return RunReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _pct(value: float | None) -> str:
    return "  -  " if value is None else f"{value:6.1%}"


def print_report(report: RunReport) -> None:
    print(f"run {report.run_id}  {report.provider}/{report.model}  dataset={report.dataset}  calls={report.n_calls}")
    print(f"overall accuracy {report.overall_accuracy:.1%}")
    print(f"{'rule':<26}{'acc':>8}{'fail P':>9}{'fail R':>9}   tp fp fn tn  err")
    for r in report.rules:
        c = r.confusion
        print(
            f"{r.rule_id:<26}{_pct(r.accuracy):>8}{_pct(r.fail_precision):>9}{_pct(r.fail_recall):>9}"
            f"   {c['tp']:>2} {c['fp']:>2} {c['fn']:>2} {c['tn']:>2}  {r.errors:>3}"
        )
    if report.total_input_tokens:
        cost = f", est. ${report.estimated_cost_usd:.4f}" if report.estimated_cost_usd is not None else ""
        print(
            f"tokens in/out {report.total_input_tokens:,}/{report.total_output_tokens:,}{cost}; "
            f"latency p50 {report.p50_latency_ms} ms, p95 {report.p95_latency_ms} ms"
        )
    if report.disagreements:
        print(f"\n{len(report.disagreements)} disagreements with the golden set:")
        for d in report.disagreements:
            print(f"  {d.case_id} / {d.rule_id}: expected {d.expected}, got {d.got or 'error'} - {d.reason}")


def cmd_run(args: argparse.Namespace) -> int:
    dataset = load_dataset(args.dataset)
    if args.judge == "replay":
        if not args.replay_file:
            print("--replay-file is required with --judge replay", file=sys.stderr)
            return 2
        judge = ReplayJudge(args.replay_file)
    else:
        judge = AnthropicJudge(model=args.model, effort=args.effort)

    report = run(dataset, judge, concurrency=args.concurrency, run_id=args.run_id)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    if args.save_predictions:
        rec = Path(args.save_predictions)
        rec.parent.mkdir(parents=True, exist_ok=True)
        rec.write_text("".join(p.model_dump_json() + "\n" for p in report.predictions), encoding="utf-8")
    print_report(report)
    print(f"\nreport written to {out}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    result = check(
        _load_report(args.baseline),
        _load_report(args.current),
        max_accuracy_drop=args.max_accuracy_drop,
        max_recall_drop=args.max_recall_drop,
        max_error_rate=args.max_error_rate,
    )
    for line in result.notes:
        print(f"  ok    {line}")
    for line in result.failures:
        print(f"  FAIL  {line}")
    print("gate passed" if result.passed else "gate FAILED")
    return 0 if result.passed else 1


def cmd_show(args: argparse.Namespace) -> int:
    print_report(_load_report(args.report))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="judge every (case, rule) pair and score it against the golden labels")
    p_run.add_argument("--dataset", required=True, help="directory with rules.json, cases.jsonl, labels.jsonl")
    p_run.add_argument("--judge", choices=["anthropic", "replay"], default="anthropic")
    p_run.add_argument("--model", default="claude-opus-5-5")
    p_run.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="medium")
    p_run.add_argument("--replay-file", help="recorded predictions (JSONL) for --judge replay")
    p_run.add_argument("--concurrency", type=int, default=4)
    p_run.add_argument("--run-id", help="label for this run (default: UTC timestamp)")
    p_run.add_argument("--out", default="results/latest.json")
    p_run.add_argument("--save-predictions", help="also write the raw predictions as JSONL, for later replay")
    p_run.set_defaults(func=cmd_run)

    p_gate = sub.add_parser("gate", help="exit 1 if the current run regressed against the baseline")
    p_gate.add_argument("--baseline", required=True)
    p_gate.add_argument("--current", required=True)
    p_gate.add_argument("--max-accuracy-drop", type=float, default=0.02)
    p_gate.add_argument("--max-recall-drop", type=float, default=0.05)
    p_gate.add_argument("--max-error-rate", type=float, default=0.0)
    p_gate.set_defaults(func=cmd_gate)

    p_show = sub.add_parser("show", help="print a saved report")
    p_show.add_argument("report")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
