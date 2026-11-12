import Link from "next/link";
import { listReports, loadCases, loadReport, pct, type RuleMetrics } from "@/lib/report";

export const dynamic = "force-dynamic";

function recallClass(r: RuleMetrics) {
  if (r.fail_recall === null) return "";
  return r.fail_recall < 0.8 ? "warn" : "ok";
}

export default async function Page({ searchParams }: { searchParams: Promise<{ run?: string }> }) {
  const { run } = await searchParams;
  const files = await listReports();
  if (files.length === 0) {
    return (
      <main>
        <h1>evalkit</h1>
        <p>No reports in <code>results/</code>. Run <code>evalkit run …</code> first.</p>
      </main>
    );
  }
  const selected = run && files.includes(run) ? run : files[0];
  const report = await loadReport(selected, files);
  if (!report) return <main><p>Report not found.</p></main>;
  const cases = await loadCases(report.dataset);
  const errors = report.rules.reduce((s, r) => s + r.errors, 0);

  return (
    <main>
      <header>
        <h1>evalkit</h1>
        <nav aria-label="Runs">
          {files.map((f) => (
            <Link key={f} href={`/?run=${encodeURIComponent(f)}`} className={f === selected ? "active" : ""}>
              {f.replace(/\.json$/, "")}
            </Link>
          ))}
        </nav>
      </header>

      <p className="meta">
        Run <b>{report.run_id}</b> · judge <b>{report.provider}/{report.model}</b> · dataset <b>{report.dataset}</b> ·{" "}
        {new Date(report.created_at).toLocaleString("en-GB", { timeZone: "UTC" })} UTC
      </p>

      <section className="tiles">
        <div><span>Overall accuracy</span><strong>{pct(report.overall_accuracy)}</strong></div>
        <div><span>Judge calls</span><strong>{report.n_calls}</strong></div>
        <div><span>Disagreements</span><strong>{report.disagreements.length}</strong></div>
        <div><span>Errored calls</span><strong>{errors}</strong></div>
        <div>
          <span>Tokens in / out</span>
          <strong>{report.total_input_tokens ? `${report.total_input_tokens.toLocaleString()} / ${report.total_output_tokens.toLocaleString()}` : "–"}</strong>
          {report.estimated_cost_usd !== null && <small>≈ ${report.estimated_cost_usd.toFixed(3)}</small>}
        </div>
      </section>

      <h2>Per rule</h2>
      <p className="hint">&ldquo;Fail&rdquo; is the positive class: recall is the share of real violations the judge caught.</p>
      <table>
        <thead>
          <tr><th>Rule</th><th>Accuracy</th><th>Fail precision</th><th>Fail recall</th><th>TP</th><th>FP</th><th>FN</th><th>TN</th><th>Errors</th></tr>
        </thead>
        <tbody>
          {report.rules.map((r) => (
            <tr key={r.rule_id}>
              <td><code>{r.rule_id}</code></td>
              <td>{pct(r.accuracy)}</td>
              <td>{pct(r.fail_precision)}</td>
              <td className={recallClass(r)}>{pct(r.fail_recall)}</td>
              <td>{r.confusion.tp}</td><td>{r.confusion.fp}</td><td>{r.confusion.fn}</td><td>{r.confusion.tn}</td>
              <td>{r.errors}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Disagreements with the golden set</h2>
      {report.disagreements.length === 0 && <p>None.</p>}
      <ul className="disagreements">
        {report.disagreements.map((d) => {
          const c = cases.get(d.case_id);
          return (
            <li key={`${d.case_id}-${d.rule_id}`}>
              <div className="row">
                <code>{d.case_id}</code> · <code>{d.rule_id}</code>
                <span className={`pill ${d.expected}`}>expected {d.expected}</span>
                <span className={`pill ${d.got ?? "error"}`}>got {d.got ?? "error"}</span>
              </div>
              <p>{d.reason}</p>
              {c && (
                <details>
                  <summary>Ticket</summary>
                  <pre>{JSON.stringify(c.data, null, 2)}</pre>
                </details>
              )}
            </li>
          );
        })}
      </ul>
      {report.provider === "replay" && report.model.startsWith("demo") && (
        <p className="hint">This run replays a hand-written demo fixture, not a live model run.</p>
      )}
    </main>
  );
}
