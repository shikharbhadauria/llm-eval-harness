from pathlib import Path

import pytest

from evalkit.cli import main
from evalkit.dataset import load_dataset
from evalkit.gate import check
from evalkit.metrics import rule_metrics
from evalkit.models import Prediction
from evalkit.providers import ReplayJudge, _verdict_schema
from evalkit.runner import build_report, run

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "examples" / "support_audit"
FIXTURE = DATASET / "fixtures" / "demo-judge.jsonl"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATASET)


def perfect_predictions(dataset):
    return [
        Prediction(case_id=c.id, rule_id=r.id, verdict=dataset.label_for(c.id, r.id).label)
        for c in dataset.cases
        for r in dataset.rules
    ]


def test_dataset_is_complete(dataset):
    assert len(dataset.cases) == 20
    assert len(dataset.rules) == 3
    assert len(dataset.labels) == 60


def test_perfect_judge_scores_100(dataset):
    report = build_report(dataset, perfect_predictions(dataset), provider="test", model="oracle")
    assert report.overall_accuracy == 1.0
    assert all(r.fail_recall == 1.0 and r.fail_precision == 1.0 for r in report.rules)
    assert report.disagreements == []


def test_errors_count_as_misses(dataset):
    preds = perfect_predictions(dataset)
    # Turn one golden failure into an errored call.
    target = next(p for p in preds if p.rule_id == "no_pii_in_public_reply" and p.verdict == "fail")
    target.verdict, target.error = None, "refusal: None"
    m = rule_metrics(dataset, preds, "no_pii_in_public_reply")
    assert m.errors == 1
    assert m.confusion["fn"] == 1
    assert m.fail_recall == 0.75


def test_replay_demo_fixture(dataset):
    report = run(dataset, ReplayJudge(FIXTURE), concurrency=2)
    assert report.n_calls == 60
    assert report.overall_accuracy == pytest.approx(57 / 60, abs=1e-4)
    by_rule = {r.rule_id: r for r in report.rules}
    assert by_rule["no_pii_in_public_reply"].fail_recall == 0.75
    assert by_rule["troubleshooting_quality"].confusion["fp"] == 1
    assert {(d.case_id, d.rule_id) for d in report.disagreements} == {
        ("case-10", "sla_first_response"),
        ("case-07", "no_pii_in_public_reply"),
        ("case-14", "troubleshooting_quality"),
    }
    assert report.estimated_cost_usd is None


def test_gate_passes_on_same_run_and_fails_on_recall_drop(dataset):
    baseline = build_report(dataset, perfect_predictions(dataset), provider="test", model="oracle")
    assert check(baseline, baseline).passed

    worse = perfect_predictions(dataset)
    for p in worse:
        if p.rule_id == "no_pii_in_public_reply" and p.verdict == "fail" and p.case_id in {"case-03", "case-07"}:
            p.verdict = "pass"
    current = build_report(dataset, worse, provider="test", model="oracle")
    result = check(baseline, current, max_accuracy_drop=0.05, max_recall_drop=0.05)
    assert not result.passed
    assert any("no_pii_in_public_reply fail-recall" in f for f in result.failures)


def test_gate_fails_on_errors(dataset):
    baseline = build_report(dataset, perfect_predictions(dataset), provider="test", model="oracle")
    preds = perfect_predictions(dataset)
    preds[0].verdict, preds[0].error = None, "api_error 529: overloaded"
    current = build_report(dataset, preds, provider="test", model="oracle")
    result = check(baseline, current, max_accuracy_drop=1.0, max_recall_drop=1.0)
    assert not result.passed
    assert any("errored" in f for f in result.failures)


def test_verdict_schema_is_strict():
    schema = _verdict_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"verdict", "reason"}
    assert schema["properties"]["verdict"]["enum"] == ["pass", "fail"]


def test_cli_run_and_gate(tmp_path, capsys):
    out = tmp_path / "run.json"
    assert main(["run", "--dataset", str(DATASET), "--judge", "replay", "--replay-file", str(FIXTURE), "--out", str(out)]) == 0
    assert main(["gate", "--baseline", str(out), "--current", str(out)]) == 0
    assert "gate passed" in capsys.readouterr().out


class _Block:
    def __init__(self, text):
        self.type, self.text = "text", text


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block(text)]
        self.stop_reason = stop_reason
        self.stop_details = None
        self.usage = type("U", (), {"input_tokens": 410, "output_tokens": 38})()


class _FakeClient:
    """Stands in for anthropic.Anthropic() and records the request it got."""

    def __init__(self, resp):
        self.resp, self.calls = resp, []
        self.beta = type("B", (), {"messages": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp


def test_anthropic_judge_request_and_parse(dataset):
    from evalkit.providers import AnthropicJudge

    fake = _FakeClient(_Resp('{"verdict": "fail", "reason": "Reply 1 contains a full API key."}'))
    judge = AnthropicJudge(client=fake)
    pred = judge.judge(dataset.cases[6], dataset.rules[1])
    assert pred.verdict == "fail" and pred.error is None
    assert pred.usage.input_tokens == 410
    req = fake.calls[0]
    assert req["model"] == "claude-opus-5-5"
    assert req["output_config"]["format"]["type"] == "json_schema"
    assert req["fallbacks"] == "default"


def test_anthropic_judge_refusal_is_an_error(dataset):
    from evalkit.providers import AnthropicJudge

    judge = AnthropicJudge(client=_FakeClient(_Resp("", stop_reason="refusal")))
    pred = judge.judge(dataset.cases[0], dataset.rules[0])
    assert pred.verdict is None and pred.error.startswith("refusal")
