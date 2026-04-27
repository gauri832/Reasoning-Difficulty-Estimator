const { useMemo, useState } = React;

function clamp(num, min, max) {
  return Math.max(min, Math.min(max, num));
}

function normalizeAnswerText(raw) {
  if (!raw) return "";

  return String(raw)
    .replace(/\r/g, "")
    .replace(/\\\(/g, "")
    .replace(/\\\)/g, "")
    .replace(/\\\[/g, "")
    .replace(/\\\]/g, "")
    .replace(/\\rvert|\\lvert/g, "|")
    .replace(/\\times/g, "x")
    .replace(/\\cdot/g, "*")
    .replace(/\\leq/g, "<=")
    .replace(/\\geq/g, ">=")
    .replace(/\\neq/g, "!=")
    .replace(/\\sqrt/g, "sqrt")
    .replace(/\\frac\{([^}]*)\}\{([^}]*)\}/g, "($1)/($2)")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/^#{1,6}\s*/gm, "")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function parseAnswerBlocks(raw) {
  const text = normalizeAnswerText(raw);
  if (!text) return [];

  return text
    .split(/\n{2,}/)
    .map((block) => block.trim())
    .filter(Boolean)
    .map((block) => {
      const lines = block.split("\n").map((line) => line.trim()).filter(Boolean);

      if (lines.length > 1 && lines.every((line) => /^\d+\.\s+/.test(line))) {
        return {
          type: "ol",
          items: lines.map((line) => line.replace(/^\d+\.\s+/, "").trim())
        };
      }

      if (lines.length > 1 && lines.every((line) => /^[-*]\s+/.test(line))) {
        return {
          type: "ul",
          items: lines.map((line) => line.replace(/^[-*]\s+/, "").trim())
        };
      }

      return {
        type: "p",
        text: lines.join(" ").replace(/\s{2,}/g, " ").trim()
      };
    });
}

function ProgressRing({ label, value, display, tone, hint }) {
  const pct = clamp(value, 0, 1) * 100;
  return (
    <div className="ring-card">
      <div
        className={`ring ring-${tone}`}
        style={{ "--progress": `${pct}%` }}
      >
        <span>{display}</span>
      </div>
      <div className="ring-meta">
        <h4>{label}</h4>
        {hint ? <p>{hint}</p> : null}
      </div>
    </div>
  );
}

function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [latest, setLatest] = useState(null);
  const [history, setHistory] = useState([]);

  const examples = [
    "What is 15 + 27?",
    "Explain the time complexity of merge sort.",
    "Prove that sqrt(2) is irrational."
  ];

  const stats = useMemo(() => {
    const totalQueries = history.length;
    const tokens = history.reduce((sum, row) => sum + Number(row.tokens || 0), 0);
    const baseline = history.reduce((sum, row) => sum + Number(row.always_cot_tokens || 0), 0);
    const saved = baseline > 0 ? ((baseline - tokens) / baseline) * 100 : 0;
    return { totalQueries, tokens, baseline, saved };
  }, [history]);

  const answerBlocks = useMemo(() => parseAnswerBlocks(latest?.answer), [latest?.answer]);

  const benchmarkMetrics = useMemo(
    () => ({
      rings: [
        { label: "Macro F1", value: 0.801, display: "0.801", tone: "violet", hint: "RDE primary metric" },
        { label: "Hard Recall", value: 0.821, display: "0.821", tone: "teal", hint: "Recall-focused safety" },
        { label: "Token Savings", value: 0.896, display: "89.6%", tone: "pink", hint: "Benchmark run" },
        { label: "Routing Accuracy", value: 1.0, display: "100%", tone: "cyan", hint: "14/14 interactive" }
      ],
      perClass: [
        { name: "Easy", precision: 0.826, recall: 0.877, f1: 0.85 },
        { name: "Medium", precision: 0.815, recall: 0.733, f1: 0.772 },
        { name: "Hard", precision: 0.744, recall: 0.821, f1: 0.78 }
      ],
      pipelineBars: [
        { label: "RES", value: 0.64, display: "> 0 (positive)", color: "teal" },
        { label: "Token Savings - Benchmark", value: 0.896, display: "89.6%", color: "cyan" },
        { label: "Token Savings - UI Session", value: 0.624, display: "62.4%", color: "violet" },
        { label: "Routing Accuracy - Benchmark", value: 1.0, display: "100%", color: "pink" },
        { label: "Routing Accuracy - Interactive", value: 1.0, display: "100%", color: "teal" }
      ],
      tokenModes: [
        { mode: "Easy / Fast", min: 12, max: 53 },
        { mode: "Medium / CoT", min: 64, max: 192 },
        { mode: "Hard / Best-of-N", min: 64, max: 320 }
      ],
      ablation: {
        v1: 0.605,
        v2: 0.78,
        delta: 0.175
      },
      summary: [
        { metric: "Test Accuracy", component: "RDE", result: "80.5%" },
        { metric: "Macro F1", component: "RDE", result: "0.801" },
        { metric: "Hard-class F1", component: "RDE", result: "0.780" },
        { metric: "Hard-class Recall", component: "RDE", result: "0.821" },
        { metric: "Separability Score (varentropy)", component: "Signal validation", result: "1.21" },
        { metric: "RES", component: "Full pipeline", result: "> 0 (positive)" },
        { metric: "Token Savings", component: "Full pipeline", result: "89.6% / 62.4%" },
        { metric: "Routing Accuracy", component: "Full pipeline", result: "100% (14/14)" },
        { metric: "Hard F1 delta (ablation)", component: "Feature importance", result: "+0.175" }
      ]
    }),
    []
  );

  async function submitQuestion() {
    const trimmed = question.trim();
    if (!trimmed || loading) return;

    setLoading(true);
    setError("");

    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed })
      });
      const data = await res.json();

      if (!res.ok) throw new Error(data.error || "Request failed");
      setLatest(data);
      setHistory((prev) => [data, ...prev]);
    } catch (e) {
      setError(e.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  }

  function confidencePct(conf) {
    return `${Math.round((conf || 0) * 100)}%`;
  }

  function niceMode(mode) {
    if (mode === "best_of_n") return "best_of_n";
    return mode || "-";
  }

  function metricNumber(v) {
    return (v || 0).toFixed(3);
  }

  function modeScale(max) {
    const globalMax = 320;
    return `${(max / globalMax) * 100}%`;
  }

  return (
    <div className="page">
      <section className="hero panel">
        <div>
          <h1>ARC Adaptive Reasoning Console</h1>
          <p>RAG-inspired operator dashboard for routing each query to the minimum reasoning budget that keeps quality intact.</p>
        </div>
        <div className="hero-score">
          <span>Primary Contribution</span>
          <strong>RES &gt; 0</strong>
          <small>DeltaAccuracy / DeltaTokens</small>
        </div>
      </section>

      <section className="top-grid">
        <article className="panel card">
          <h2>Ask Question</h2>
          <textarea
            placeholder="Type a question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
          />
          <button onClick={submitQuestion} disabled={loading}>
            {loading ? "Running..." : "Submit"}
          </button>

          <div className="examples">
            {examples.map((x) => (
              <span key={x} className="chip" onClick={() => setQuestion(x)}>
                {x}
              </span>
            ))}
          </div>
          {error && <div className="status warn">{error}</div>}
        </article>

        <article className="panel card">
          <h2>ARC Trace</h2>
          {latest ? (
            <>
              <div className="trace-row">
                <span className="label">Difficulty</span>
                <strong>{latest.difficulty.toUpperCase()}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Confidence</span>
                <strong>{confidencePct(latest.rde_conf)}</strong>
                <div className="confidence">
                  <div style={{ width: confidencePct(latest.rde_conf) }}></div>
                </div>
              </div>
              <div className="trace-row">
                <span className="label">Mode</span>
                <strong>{niceMode(latest.mode)}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Escalated</span>
                <strong>{latest.escalated ? "Yes" : "No"}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Reason</span>
                <span>{latest.mode_reason}</span>
              </div>
              <div className="trace-row">
                <span className="label">Tokens Used</span>
                <strong>{latest.tokens}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Always-CoT</span>
                <strong>{latest.always_cot_tokens}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Query Savings</span>
                <strong className={latest.query_savings_pct >= 0 ? "good" : "warn"}>
                  {latest.query_savings_pct.toFixed(1)}%
                </strong>
              </div>
            </>
          ) : (
            <div className="status">Submit a question to view routing trace.</div>
          )}
        </article>
      </section>

      <section className="panel card answer-block">
        <h2>Answer</h2>
        {latest ? (
          <div className="answer">
            {answerBlocks.length ? (
              answerBlocks.map((block, idx) => {
                if (block.type === "ol") {
                  return (
                    <ol key={idx}>
                      {block.items.map((item, itemIdx) => (
                        <li key={itemIdx}>{item}</li>
                      ))}
                    </ol>
                  );
                }

                if (block.type === "ul") {
                  return (
                    <ul key={idx}>
                      {block.items.map((item, itemIdx) => (
                        <li key={itemIdx}>{item}</li>
                      ))}
                    </ul>
                  );
                }

                return <p key={idx}>{block.text}</p>;
              })
            ) : (
              <p>(No answer generated)</p>
            )}
          </div>
        ) : (
          <div className="status">Answer will appear here.</div>
        )}
      </section>

      <section className="panel card session">
        <h2>Session Stats</h2>
        <div className="session-grid">
          <div className="metric">
            <div className="k">Queries</div>
            <div className="v">{stats.totalQueries}</div>
          </div>
          <div className="metric">
            <div className="k">ARC Tokens</div>
            <div className="v">{stats.tokens}</div>
          </div>
          <div className="metric">
            <div className="k">Always-CoT Tokens</div>
            <div className="v">{stats.baseline}</div>
          </div>
          <div className="metric">
            <div className="k">Cumulative Savings</div>
            <div className={`v ${stats.saved >= 0 ? "good" : "warn"}`}>{stats.saved.toFixed(1)}%</div>
          </div>
        </div>

        <div className="history">
          {history.map((row, idx) => (
            <div className="history-item" key={`${row.question}-${idx}`}>
              <strong>Q:</strong> {row.question} | <strong>{row.mode}</strong> | {row.tokens} tok | {row.query_savings_pct.toFixed(1)}%
            </div>
          ))}
        </div>
      </section>

      <section className="perf">
        <div className="section-head">
          <h2>Performance Metrics</h2>
          <p>Visual summary of classifier quality, routing behavior, and compute-efficiency gains.</p>
        </div>

        <div className="ring-grid">
          {benchmarkMetrics.rings.map((item) => (
            <ProgressRing
              key={item.label}
              label={item.label}
              value={item.value}
              display={item.display}
              tone={item.tone}
              hint={item.hint}
            />
          ))}
        </div>

        <div className="chart-grid">
          <article className="panel card chart-card">
            <h3>Per-Class Precision / Recall / F1</h3>
            <div className="legend">
              <span className="swatch swatch-precision">Precision</span>
              <span className="swatch swatch-recall">Recall</span>
              <span className="swatch swatch-f1">F1</span>
            </div>
            {benchmarkMetrics.perClass.map((row) => (
              <div className="class-row" key={row.name}>
                <div className="class-name">{row.name}</div>

                <div className="metric-track">
                  <div className="metric-fill metric-fill-precision" style={{ width: `${row.precision * 100}%` }}></div>
                </div>
                <div className="metric-value">{metricNumber(row.precision)}</div>

                <div className="metric-track">
                  <div className="metric-fill metric-fill-recall" style={{ width: `${row.recall * 100}%` }}></div>
                </div>
                <div className="metric-value">{metricNumber(row.recall)}</div>

                <div className="metric-track">
                  <div className="metric-fill metric-fill-f1" style={{ width: `${row.f1 * 100}%` }}></div>
                </div>
                <div className="metric-value">{metricNumber(row.f1)}</div>
              </div>
            ))}
          </article>

          <article className="panel card chart-card">
            <h3>Pipeline Efficiency Snapshot</h3>
            {benchmarkMetrics.pipelineBars.map((bar) => (
              <div className="pipeline-row" key={bar.label}>
                <div className="pipeline-label">{bar.label}</div>
                <div className="pipeline-track">
                  <div className={`pipeline-fill pipeline-fill-${bar.color}`} style={{ width: `${clamp(bar.value, 0, 1) * 100}%` }}></div>
                </div>
                <div className="pipeline-value">{bar.display}</div>
              </div>
            ))}

            <h4 className="sub-head">Token Range By Mode</h4>
            {benchmarkMetrics.tokenModes.map((row) => (
              <div className="token-row" key={row.mode}>
                <span>{row.mode}</span>
                <div className="token-track">
                  <div className="token-min" style={{ left: modeScale(row.min) }}></div>
                  <div className="token-max" style={{ width: modeScale(row.max) }}></div>
                </div>
                <strong>{row.min}-{row.max}</strong>
              </div>
            ))}
          </article>
        </div>

        <div className="chart-grid">
          <article className="panel card chart-card">
            <h3>Hard-Class Ablation (v1 vs v2)</h3>
            <div className="ablation-bars">
              <div className="ablation-col">
                <div className="ablation-bar ablation-v1" style={{ height: `${benchmarkMetrics.ablation.v1 * 100}%` }}></div>
                <span>v1 (0.605)</span>
              </div>
              <div className="ablation-col">
                <div className="ablation-bar ablation-v2" style={{ height: `${benchmarkMetrics.ablation.v2 * 100}%` }}></div>
                <span>v2 (0.780)</span>
              </div>
            </div>
            <div className="delta-chip">Hard F1 Delta +{benchmarkMetrics.ablation.delta.toFixed(3)}</div>
          </article>

          <article className="panel card chart-card">
            <h3>Summary Matrix</h3>
            <div className="perf-table">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Component</th>
                    <th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {benchmarkMetrics.summary.map((row) => (
                    <tr key={row.metric}>
                      <td>{row.metric}</td>
                      <td>{row.component}</td>
                      <td className={row.metric === "RES" ? "accent" : ""}>{row.result}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </article>
        </div>
      </section>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
