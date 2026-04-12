import re
from typing import Dict

import numpy as np
import pandas as pd

INPUT_PATH = "data/features/signals_v2_flat.csv"
OUTPUT_PATH = "data/features/labeled_signals_v2.csv"

EASY_SOURCES = {"gsm8k", "mbpp"}
MEDIUM_SOURCES = {"mmlu", "bbh"}
HARD_SOURCES = {"math_algebra", "math_geometry", "math_number_theory"}

HARD_CUE_RE = re.compile(
    r"\b(prove|proof|contradiction|induction|lemma|theorem|corollary|"
    r"irrational|integer solutions?|recurrence|derive|show that|if and only if|iff)\b",
    re.IGNORECASE,
)
MEDIUM_CUE_RE = re.compile(
    r"\b(explain|compare|difference|complexity|design|implement|algorithm|"
    r"analyze|average speed|time complexity|queue|stack|binary search)\b",
    re.IGNORECASE,
)
EASY_ARITH_RE = re.compile(
    r"^\s*(what is|calculate|compute|find)\b.{0,80}(\d+\s*[+\-*/]\s*\d+)",
    re.IGNORECASE,
)


def _z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mu = values.mean()
    sigma = values.std(ddof=0)
    if not np.isfinite(sigma) or sigma < 1e-8:
        return pd.Series(np.zeros(len(values), dtype=np.float32), index=series.index)
    return ((values - mu) / sigma).fillna(0.0).astype(np.float32)


def _source_prior(source: str) -> str:
    if source in EASY_SOURCES:
        return "easy"
    if source in MEDIUM_SOURCES:
        return "medium"
    if source in HARD_SOURCES:
        return "hard"
    return "medium"


def _complexity_score(df: pd.DataFrame) -> pd.Series:
    feat = {
        "attn_entropy": _z(df["attn_entropy"]),
        "varentropy": _z(df["varentropy"]),
        "perplexity": _z(df["perplexity"]),
        "tree_depth": _z(df["tree_depth"]),
        "clause_count": _z(df["clause_count"]),
        "avg_sent_len": _z(df["avg_sent_len"]),
        "proof_kw_density": _z(df["proof_kw_density"]),
        "math_sym_density": _z(df["math_sym_density"]),
        "eq_density": _z(df["eq_density"]),
        "abstract_ratio": _z(df["abstract_ratio"]),
        "avg_word_len": _z(df["avg_word_len"]),
        "num_density": _z(df["num_density"]),
    }

    # Positive weights for structural/formal complexity; negative for pure arithmetic signal.
    score = (
        0.12 * feat["attn_entropy"]
        + 0.20 * feat["varentropy"]
        - 0.08 * feat["perplexity"]
        + 0.13 * feat["tree_depth"]
        + 0.10 * feat["clause_count"]
        + 0.12 * feat["avg_sent_len"]
        + 0.25 * feat["proof_kw_density"]
        + 0.22 * feat["math_sym_density"]
        + 0.18 * feat["eq_density"]
        + 0.16 * feat["abstract_ratio"]
        + 0.08 * feat["avg_word_len"]
        - 0.16 * feat["num_density"]
    )
    return score.astype(np.float32)


def _rebalance(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    hard_target = int(0.15 * n)
    easy_cap = int(0.52 * n)

    current = df["difficulty"].value_counts()
    hard_now = int(current.get("hard", 0))
    easy_now = int(current.get("easy", 0))

    if hard_now < hard_target:
        need = hard_target - hard_now
        promote = (
            df[df["difficulty"] == "medium"]
            .sort_values("complexity_score", ascending=False)
            .head(need)
            .index
        )
        df.loc[promote, "difficulty"] = "hard"

    if easy_now > easy_cap:
        need = easy_now - easy_cap
        promote = (
            df[df["difficulty"] == "easy"]
            .sort_values("complexity_score", ascending=False)
            .head(need)
            .index
        )
        df.loc[promote, "difficulty"] = "medium"

    return df


def assign_label(row: pd.Series, thr: Dict[str, float]) -> str:
    source = row["source"]
    text = str(row.get("question", ""))
    score = float(row["complexity_score"])

    hard_cue = bool(HARD_CUE_RE.search(text))
    medium_cue = bool(MEDIUM_CUE_RE.search(text))
    easy_arith = bool(EASY_ARITH_RE.search(text))

    label = _source_prior(source)

    if hard_cue or score >= thr["hard_high"]:
        label = "hard"
    elif score >= thr["medium_high"] or medium_cue:
        label = "medium"
    else:
        label = "easy"

    # Source priors as soft guards, not hard constraints.
    if source in HARD_SOURCES and label == "easy":
        label = "medium"
    if source in MEDIUM_SOURCES and label == "easy" and score > thr["easy_low"]:
        label = "medium"

    # Keep short arithmetic prompts easy unless explicit medium/hard cues exist.
    if easy_arith and not medium_cue and not hard_cue and score < thr["medium_high"]:
        label = "easy"

    return label


def main() -> None:
    df = pd.read_csv(INPUT_PATH)
    print("Sources:\n")
    print(df["source"].value_counts())

    df = df.copy()
    df["complexity_score"] = _complexity_score(df)

    thr = {
        "easy_low": float(df["complexity_score"].quantile(0.28)),
        "medium_high": float(df["complexity_score"].quantile(0.56)),
        "hard_high": float(df["complexity_score"].quantile(0.82)),
    }

    df["difficulty"] = df.apply(lambda row: assign_label(row, thr), axis=1)
    df = _rebalance(df)

    # Keep output schema expected by downstream scripts.
    df.drop(columns=["complexity_score"], inplace=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nFinal distribution:")
    print(df["difficulty"].value_counts())
    print(df["difficulty"].value_counts(normalize=True).round(3))

    print("\nSource x difficulty:")
    print(pd.crosstab(df["source"], df["difficulty"]))


if __name__ == "__main__":
    main()
