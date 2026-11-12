import { promises as fs } from "node:fs";
import path from "node:path";

// Mirrors evalkit/models.py (RunReport and friends).
export type Label = "pass" | "fail";

export interface RuleMetrics {
  rule_id: string;
  n: number;
  accuracy: number;
  fail_precision: number | null;
  fail_recall: number | null;
  confusion: { tp: number; fp: number; fn: number; tn: number };
  errors: number;
}

export interface Disagreement {
  case_id: string;
  rule_id: string;
  expected: Label;
  got: Label | null;
  reason: string;
}

export interface RunReport {
  run_id: string;
  provider: string;
  model: string;
  dataset: string;
  created_at: string;
  n_calls: number;
  overall_accuracy: number;
  rules: RuleMetrics[];
  disagreements: Disagreement[];
  total_input_tokens: number;
  total_output_tokens: number;
  estimated_cost_usd: number | null;
  p50_latency_ms: number;
  p95_latency_ms: number;
}

export interface Case {
  id: string;
  data: Record<string, unknown>;
}

const REPO_ROOT = path.resolve(process.cwd(), "..");
const RESULTS_DIR = process.env.EVALKIT_RESULTS_DIR ?? path.join(REPO_ROOT, "results");

/** Report files in results/, newest first. */
export async function listReports(): Promise<string[]> {
  const entries = await fs.readdir(RESULTS_DIR).catch(() => [] as string[]);
  const files = entries.filter((f) => f.endsWith(".json"));
  const stats = await Promise.all(files.map(async (f) => ({ f, t: (await fs.stat(path.join(RESULTS_DIR, f))).mtimeMs })));
  return stats.sort((a, b) => b.t - a.t).map((s) => s.f);
}

/** Loads a report by file name. Only names returned by listReports() are accepted. */
export async function loadReport(file: string, allowed: string[]): Promise<RunReport | null> {
  if (!allowed.includes(file)) return null;
  const raw = await fs.readFile(path.join(RESULTS_DIR, file), "utf-8");
  return JSON.parse(raw) as RunReport;
}

/** Cases for the report's dataset, keyed by id, so disagreements can show the ticket. */
export async function loadCases(dataset: string): Promise<Map<string, Case>> {
  if (!/^[\w-]+$/.test(dataset)) return new Map();
  const file = path.join(REPO_ROOT, "examples", dataset, "cases.jsonl");
  const raw = await fs.readFile(file, "utf-8").catch(() => "");
  const cases = raw
    .split("\n")
    .filter((l) => l.trim())
    .map((l) => JSON.parse(l) as Case);
  return new Map(cases.map((c) => [c.id, c]));
}

export const pct = (v: number | null) => (v === null ? "–" : `${(v * 100).toFixed(1)}%`);
