# llm-eval-harness

Regression evaluation for **rule-based LLM judges**: systems where an LLM checks each item (a support ticket, a document, a transcript) against a list of rules, one call per rule.

You keep a small **golden dataset** of hand-reviewed verdicts. Every change to a prompt, model or rule is scored against it, and a **CI gate** blocks the change if accuracy or recall drops. A Next.js page shows each run and where the judge disagreed with the golden labels.

```
cases.jsonl + rules.json ──► judge (Claude, structured output) ──► predictions
                                                                    │
labels.jsonl (golden, hand-reviewed) ──────────────────────────────►├─► metrics per rule
                                                                    ├─► disagreements
                                                                    └─► gate vs baseline ──► pass / fail (CI)
```

## What it measures

| Metric | Why |
|---|---|
| Accuracy per rule and overall | The headline number |
| **Fail recall** per rule | The share of real violations the judge caught. In an audit a missed violation costs more than a false alarm, so the gate has a separate limit for recall |
| Fail precision per rule | How often a "fail" verdict is right; this drives reviewer workload |
| Confusion counts (TP/FP/FN/TN) | To see *how* a rule is wrong |
| Errored calls | Refusals, rate limits and truncation count as misses and fail the gate by default |
| Tokens, estimated cost, p50/p95 latency | Cost and speed per run, so a cheaper model or lower effort can be compared on equal terms |

## Quick start

```bash
uv venv && uv pip install -e ".[dev]"
.venv/bin/pytest -q

# Replay the demo fixture (no API key needed)
.venv/bin/evalkit run --dataset examples/support_audit --judge replay \
  --replay-file examples/support_audit/fixtures/demo-judge.jsonl --out results/current.json

# Regression gate: exits 1 if the run regressed against the baseline
.venv/bin/evalkit gate --baseline results/baseline.json --current results/current.json
```

Live run against Claude (needs `ANTHROPIC_API_KEY`; 60 calls):

```bash
.venv/bin/evalkit run --dataset examples/support_audit --judge anthropic \
  --model claude-opus-5-5 --effort medium \
  --out results/live.json --save-predictions results/live.jsonl
```

`--save-predictions` records the verdicts so the run can be replayed later without calling the API.

Results UI:

```bash
cd web && npm install && npm run dev   # http://localhost:3000, reads ../results/*.json
```

## How the judge works

- **One call per rule per case.** Each rule gets its own focused prompt, so a failure can be traced to one rule and rules can be tuned independently.
- **Structured output.** The response is constrained to a JSON schema (`{"verdict": "pass" | "fail", "reason": "..."}`), so there's no free-text parsing.
- **Errors are results, not crashes.** A refusal, rate limit or truncated answer is recorded on the prediction and counted against the run.
- **Refusal fallback.** Server-side fallbacks are on, so if a safety classifier declines, the API retries on a fallback model.

## The example dataset

`examples/support_audit` has 20 **synthetic** support tickets and 3 rules:

| Rule | Fails when |
|---|---|
| `sla_first_response` | The first engineer response came after the priority's SLA. Auto-acknowledgements don't count |
| `no_pii_in_public_reply` | A public reply exposes another customer's contact details, a full card number, or any unmasked password, key or token |
| `troubleshooting_quality` | The ticket was closed without diagnostic evidence, or without verifying the fix |

There are 60 golden labels, one per (ticket, rule). The tickets are invented for this repo.

`fixtures/demo-judge.jsonl` is a **hand-written** set of verdicts with three realistic mistakes. For example, the judge counts an auto-acknowledgement as the first response. It exists so that the tests, the CI gate and the UI work without an API key. It is not output from a model, and the UI labels it as such.

## Using it on your own data

1. Create `rules.json`: `[{"id", "name", "instruction"}]`.
2. Create `cases.jsonl` with one `{"id", "data": {...}}` per line. `data` can be any JSON.
3. Create `labels.jsonl` with one `{"case_id", "rule_id", "label": "pass" | "fail"}` per line. Every (case, rule) pair needs a label; the loader rejects gaps.
4. Run once and review the disagreements. Fix the labels or the rule wording, then save that run as `results/baseline.json`.
5. From then on, gate every change against the baseline.

## CI

`.github/workflows/eval.yml` runs the unit tests and a replay run gated against `results/baseline.json` on every push. The optional live job (manual dispatch, `ANTHROPIC_API_KEY` secret) runs the real judge, gates it, and uploads the results.

## Layout

```
evalkit/      dataset loader, judges (Anthropic + replay), runner, metrics, gate, CLI
examples/     support_audit dataset and the demo fixture
results/      run reports (baseline.json is committed)
tests/        unit tests, including the request the Anthropic judge sends
web/          Next.js results viewer (TypeScript)
```
