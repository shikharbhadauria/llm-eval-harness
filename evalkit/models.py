"""Data shapes shared by the dataset loader, providers, runner and metrics."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Label = Literal["pass", "fail"]


class Rule(BaseModel):
    """One audit rule. The judge is asked about a single rule per call."""

    id: str
    name: str
    instruction: str = Field(description="What the judge must check, written for the model")


class Case(BaseModel):
    """One item under evaluation, e.g. a closed support ticket."""

    id: str
    data: dict


class GoldenLabel(BaseModel):
    """The hand-reviewed verdict for one (case, rule) pair: the source of truth."""

    case_id: str
    rule_id: str
    label: Label


class JudgeVerdict(BaseModel):
    """What the model must return. Sent to the API as a strict JSON schema."""

    model_config = ConfigDict(extra="forbid")

    verdict: Label
    reason: str = Field(description="One or two sentences citing the evidence in the case")


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class Prediction(BaseModel):
    """One judge call's result, as recorded by the runner."""

    case_id: str
    rule_id: str
    verdict: Label | None = Field(default=None, description="None when the call failed or was refused")
    reason: str = ""
    error: str | None = None
    latency_ms: int = 0
    usage: Usage = Usage()


class RuleMetrics(BaseModel):
    rule_id: str
    n: int
    accuracy: float
    fail_precision: float | None = Field(description="Of the cases the judge failed, the share the golden set also failed")
    fail_recall: float | None = Field(description="Of the golden failures, the share the judge caught")
    confusion: dict[str, int] = Field(description="tp/fp/fn/tn with 'fail' as the positive class")
    errors: int


class Disagreement(BaseModel):
    case_id: str
    rule_id: str
    expected: Label
    got: Label | None
    reason: str


class RunReport(BaseModel):
    """Everything one run produced. Saved as JSON; the gate and the UI read this."""

    run_id: str
    provider: str
    model: str
    dataset: str
    created_at: str
    n_calls: int
    overall_accuracy: float
    rules: list[RuleMetrics]
    disagreements: list[Disagreement]
    total_input_tokens: int
    total_output_tokens: int
    estimated_cost_usd: float | None
    p50_latency_ms: int
    p95_latency_ms: int
    predictions: list[Prediction]
