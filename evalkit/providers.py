"""Judges that turn (case, rule) into a verdict.

- AnthropicJudge calls Claude with a strict JSON-schema output, one call per rule per case.
- ReplayJudge reads verdicts recorded earlier, so CI can run the gate without an API key.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Protocol

from .models import Case, JudgeVerdict, Prediction, Rule, Usage

SYSTEM_PROMPT = """You audit closed customer-support tickets against one rule at a time.

You get the rule and the ticket as JSON. Decide whether the ticket passes or fails the rule, \
based only on what the ticket shows. If the ticket lacks the evidence the rule needs, judge \
from what is there and say what was missing in the reason. Keep the reason to one or two \
sentences and quote or point to the specific part of the ticket you relied on."""

# USD per million tokens (input, output). Used only to estimate run cost.
PRICES_PER_MTOK = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class Judge(Protocol):
    name: str
    model: str

    def judge(self, case: Case, rule: Rule) -> Prediction: ...


def build_user_message(case: Case, rule: Rule) -> str:
    return (
        f"Rule {rule.id}: {rule.name}\n{rule.instruction}\n\n"
        f"Ticket {case.id}:\n{json.dumps(case.data, indent=2, ensure_ascii=False)}"
    )


def _verdict_schema() -> dict:
    schema = JudgeVerdict.model_json_schema()
    schema["additionalProperties"] = False
    return schema


class AnthropicJudge:
    """One Claude call per (case, rule), returning a schema-validated verdict."""

    name = "anthropic"

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "medium", client=None):
        import anthropic

        self.model = model
        self.effort = effort
        self._anthropic = anthropic
        # Resolves ANTHROPIC_API_KEY (or an `ant auth login` profile) from the environment.
        self.client = client or anthropic.Anthropic()

    def judge(self, case: Case, rule: Rule) -> Prediction:
        anthropic = self._anthropic
        started = time.perf_counter()
        base = Prediction(case_id=case.id, rule_id=rule.id)
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(case, rule)}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": _verdict_schema()},
                },
                # If a safety classifier declines, the API re-runs the request on a fallback model.
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except anthropic.RateLimitError as exc:
            return base.model_copy(update={"error": f"rate_limited: {exc.message}", "latency_ms": _ms(started)})
        except anthropic.APIStatusError as exc:
            return base.model_copy(update={"error": f"api_error {exc.status_code}: {exc.message}", "latency_ms": _ms(started)})
        except anthropic.APIConnectionError as exc:
            return base.model_copy(update={"error": f"connection_error: {exc}", "latency_ms": _ms(started)})

        usage = Usage(input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens)
        common = {"latency_ms": _ms(started), "usage": usage}
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None) if response.stop_details else None
            return base.model_copy(update={"error": f"refusal: {category}", **common})
        if response.stop_reason == "max_tokens":
            return base.model_copy(update={"error": "max_tokens: output truncated", **common})

        text = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            return base.model_copy(update={"error": "no text block in response", **common})
        verdict = JudgeVerdict.model_validate_json(text)
        return base.model_copy(update={"verdict": verdict.verdict, "reason": verdict.reason, **common})


class ReplayJudge:
    """Returns verdicts recorded in a JSONL file (one Prediction per line)."""

    name = "replay"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.model = self.path.stem
        self._rows: dict[tuple[str, str], Prediction] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                p = Prediction.model_validate_json(line)
                self._rows[(p.case_id, p.rule_id)] = p

    def judge(self, case: Case, rule: Rule) -> Prediction:
        recorded = self._rows.get((case.id, rule.id))
        if recorded is None:
            return Prediction(case_id=case.id, rule_id=rule.id, error="no recorded verdict")
        return recorded


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    return round(input_tokens / 1e6 * prices[0] + output_tokens / 1e6 * prices[1], 4)


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
