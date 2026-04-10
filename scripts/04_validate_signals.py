# scripts/04_validate_signals.py
# Confirms: (1) class balance, (2) signal separability per class,
# (3) low inter-signal correlation. All needed before RDE training.

import pandas as pd
import numpy as np

df = pd.read_csv("data/features/labeled_signals.csv")
SIGNALS = ["attn_entropy", "varentropy", "perplexity", "tree_depth", "clause_count"]

# ── 1. Class balance ──────────────────────────────────────
print("=" * 50)
print("CLASS DISTRIBUTION")
print("=" * 50)
counts = df["difficulty"].value_counts()
pcts   = df["difficulty"].value_counts(normalize=True)
for cls in ["easy", "medium", "hard"]:
    n   = counts.get(cls, 0)
    pct = pcts.get(cls, 0)
    ok  = "✅" if pct >= 0.15 else "❌"
    print(f"  {ok} {cls:8s}: {n:4d} samples ({pct:.1%})")

# ── 2. Signal means per class ─────────────────────────────
print("\n" + "=" * 50)
print("SIGNAL MEANS PER DIFFICULTY CLASS")
print("(hard should be highest for varentropy & perplexity)")
print("=" * 50)
means = df.groupby("difficulty")[SIGNALS].mean()
print(means.round(4).to_string())

# ── 3. Separability score (F-statistic proxy) ─────────────
print("\n" + "=" * 50)
print("SIGNAL SEPARABILITY (higher = better for RDE)")
print("=" * 50)
for sig in SIGNALS:
    group_means = df.groupby("difficulty")[sig].mean()
    group_vars  = df.groupby("difficulty")[sig].var()
    overall_mean = df[sig].mean()
    # Between-class variance / within-class variance
    between = np.mean((group_means - overall_mean) ** 2)
    within  = group_vars.mean()
    score   = between / (within + 1e-9)
    ok = "✅" if score > 0.01 else "⚠️ "
    print(f"  {ok} {sig:20s}: separability score = {score:.4f}")

# ── 4. Correlation matrix ─────────────────────────────────
print("\n" + "=" * 50)
print("CORRELATION MATRIX (values <0.85 are acceptable)")
print("=" * 50)
corr = df[SIGNALS].corr().round(3)
print(corr.to_string())

high_corr = []
for i in range(len(SIGNALS)):
    for j in range(i+1, len(SIGNALS)):
        c = abs(corr.iloc[i, j])
        if c > 0.85:
            high_corr.append((SIGNALS[i], SIGNALS[j], c))

if high_corr:
    print("\n⚠️  Highly correlated pairs (consider dropping one):")
    for a, b, c in high_corr:
        print(f"   {a} ↔ {b}: {c:.3f}")
else:
    print("\n✅ No highly correlated signal pairs — all 5 signals are informative.")

# ── 5. Per-source breakdown ───────────────────────────────
print("\n" + "=" * 50)
print("SOURCE → DIFFICULTY MAPPING (sanity check)")
print("=" * 50)
print(df.groupby(["source", "difficulty"]).size().to_string())