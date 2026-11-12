"""Load a dataset directory: rules.json, cases.jsonl and labels.jsonl."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .models import Case, GoldenLabel, Rule


@dataclass
class Dataset:
    name: str
    rules: list[Rule]
    cases: list[Case]
    labels: dict[tuple[str, str], GoldenLabel]

    def label_for(self, case_id: str, rule_id: str) -> GoldenLabel:
        return self.labels[(case_id, rule_id)]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
    return rows


def load_dataset(directory: str | Path) -> Dataset:
    root = Path(directory)
    rules = [Rule.model_validate(r) for r in json.loads((root / "rules.json").read_text(encoding="utf-8"))]
    cases = [Case.model_validate(c) for c in _read_jsonl(root / "cases.jsonl")]
    labels = {}
    for row in _read_jsonl(root / "labels.jsonl"):
        label = GoldenLabel.model_validate(row)
        key = (label.case_id, label.rule_id)
        if key in labels:
            raise ValueError(f"duplicate golden label for case {label.case_id!r}, rule {label.rule_id!r}")
        labels[key] = label

    rule_ids = {r.id for r in rules}
    case_ids = {c.id for c in cases}
    if len(rule_ids) != len(rules):
        raise ValueError("rule ids must be unique")
    if len(case_ids) != len(cases):
        raise ValueError("case ids must be unique")

    missing = [(c, r) for c in sorted(case_ids) for r in sorted(rule_ids) if (c, r) not in labels]
    if missing:
        raise ValueError(f"{len(missing)} (case, rule) pairs have no golden label, e.g. {missing[0]}")
    unknown = [k for k in labels if k[0] not in case_ids or k[1] not in rule_ids]
    if unknown:
        raise ValueError(f"golden labels reference unknown cases or rules, e.g. {unknown[0]}")

    return Dataset(name=root.name, rules=rules, cases=cases, labels=labels)
