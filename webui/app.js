const { useMemo, useState } = React;

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
    const tokens = history.reduce((sum, row) => sum + row.tokens, 0);
    const baseline = history.reduce((sum, row) => sum + row.always_cot_tokens, 0);
    const saved = baseline > 0 ? ((baseline - tokens) / baseline) * 100 : 0;
    return { totalQueries, tokens, baseline, saved };
  }, [history]);

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

  return (
    <div className="page">
      <section className="hero">
        <h1>ARC - Adaptive Reasoning Controller</h1>
        <p>Route each question to the minimum reasoning budget that preserves answer quality.</p>
      </section>

      <section className="grid">
        <article className="card">
          <h2>Ask A Question</h2>
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

        <article className="card">
          <h2>ARC Trace</h2>
          {latest ? (
            <>
              <div className="trace-row">
                <span className="label">Difficulty:</span>
                <strong>{latest.difficulty.toUpperCase()}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Confidence:</span>
                <strong>{confidencePct(latest.rde_conf)}</strong>
                <div className="confidence">
                  <div style={{ width: confidencePct(latest.rde_conf) }}></div>
                </div>
              </div>
              <div className="trace-row">
                <span className="label">Mode:</span>
                <strong>{niceMode(latest.mode)}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Escalated:</span>
                <strong>{latest.escalated ? "Yes" : "No"}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Reason:</span>
                <span>{latest.mode_reason}</span>
              </div>
              <div className="trace-row">
                <span className="label">Tokens Used:</span>
                <strong>{latest.tokens}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Always-CoT:</span>
                <strong>{latest.always_cot_tokens}</strong>
              </div>
              <div className="trace-row">
                <span className="label">Query Savings:</span>
                <strong className={latest.query_savings_pct >= 0 ? "good" : "warn"}>
                  {latest.query_savings_pct.toFixed(1)}%
                </strong>
              </div>
            </>
          ) : (
            <div className="status">Submit a question to view routing trace.</div>
          )}
        </article>

        <article className="card">
          <h2>Answer</h2>
          {latest ? (
            <div className="answer">{latest.answer || "(No answer generated)"}</div>
          ) : (
            <div className="status">Answer will appear here.</div>
          )}
        </article>
      </section>

      <section className="session">
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
            <div className={`v ${stats.saved >= 0 ? "good" : "warn"}`}>
              {stats.saved.toFixed(1)}%
            </div>
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
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<App />);
