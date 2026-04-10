# scripts/03_label_difficulty.py
# FIX: Source names now match exactly what 01_download_data.py uses.
#      math_* sources are labeled hard by default, then refined
#      by varentropy percentile within each math sub-domain.
"""
import pandas as pd

df = pd.read_csv("data/features/signals_flat.csv")

print("Sources in signals file:")
print(df["source"].value_counts())
print()

# ── Step 1: Tier-based labeling using EXACT source names ──
EASY_SOURCES   = {"gsm8k", "mbpp"}
MEDIUM_SOURCES = {"mmlu", "bbh"}
HARD_SOURCES   = {"math_algebra", "math_geometry", "math_number_theory"}

def assign_tier(row):
    src = row["source"]
    if src in EASY_SOURCES:
        return "easy"
    elif src in MEDIUM_SOURCES:
        return "medium"
    elif src in HARD_SOURCES:
        return "hard"
    return "medium"   # safe fallback

df["difficulty"] = df.apply(assign_tier, axis=1)

print("After tier labeling:")
print(df["difficulty"].value_counts())
print()

# ── Step 2: Refine hard samples using varentropy percentiles ──
# Split the hard math samples into easy/medium/hard thirds
# based on varentropy (higher varentropy = more uncertain = harder)
hard_mask = df["source"].isin(HARD_SOURCES)

if hard_mask.sum() > 0:
    q33 = df.loc[hard_mask, "varentropy"].quantile(0.33)
    q66 = df.loc[hard_mask, "varentropy"].quantile(0.66)
    print(f"Math varentropy thresholds: q33={q33:.4f}, q66={q66:.4f}")

    def refine_hard(row):
        if row["source"] not in HARD_SOURCES:
            return row["difficulty"]
        if row["varentropy"] < q33:
            return "easy"
        elif row["varentropy"] < q66:
            return "medium"
        else:
            return "hard"

    df["difficulty"] = df.apply(refine_hard, axis=1)

# ── Step 3: Also refine easy sources using varentropy ─────
# gsm8k has varying complexity — top 20% by varentropy → medium
easy_mask = df["source"].isin(EASY_SOURCES)
if easy_mask.sum() > 0:
    q80_easy = df.loc[easy_mask, "varentropy"].quantile(0.80)
    def refine_easy(row):
        if row["source"] not in EASY_SOURCES:
            return row["difficulty"]
        return "medium" if row["varentropy"] > q80_easy else "easy"

    df["difficulty"] = df.apply(refine_easy, axis=1)

# ── Save ──────────────────────────────────────────────────
df.to_csv("data/features/labeled_signals.csv", index=False)

print("\n✅ Final difficulty distribution:")
print(df["difficulty"].value_counts())
print()
print("Distribution by percentage:")
print(df["difficulty"].value_counts(normalize=True).round(3))
print()

# ── Warning check ─────────────────────────────────────────
counts = df["difficulty"].value_counts(normalize=True)
for cls in ["easy", "medium", "hard"]:
    pct = counts.get(cls, 0)
    if pct < 0.15:
        print(f"⚠️  WARNING: '{cls}' is only {pct:.1%} of data — may hurt RDE training!")
    else:
        print(f"✅ '{cls}': {pct:.1%} — OK")

"""
"""
# scripts/03_label_difficulty.py
# FIX v2: Uses 50/75 percentile split on math varentropy
#          so hard class reaches ~20-25% of total data.
#          gsm8k: only top 15% promoted to medium (not hard).

import pandas as pd

df = pd.read_csv("data/features/signals_flat.csv")

print("Sources in signals file:")
print(df["source"].value_counts())
print()

EASY_SOURCES   = {"gsm8k", "mbpp"}
MEDIUM_SOURCES = {"mmlu", "bbh"}
HARD_SOURCES   = {"math_algebra", "math_geometry", "math_number_theory"}

# ── Step 1: Base tier labels ──────────────────────────────
def assign_tier(row):
    src = row["source"]
    if src in EASY_SOURCES:   return "easy"
    if src in MEDIUM_SOURCES: return "medium"
    if src in HARD_SOURCES:   return "hard"
    return "medium"

df["difficulty"] = df.apply(assign_tier, axis=1)
print("After base tier labeling:")
print(df["difficulty"].value_counts())
print()

# ── Step 2: Refine math using 50/75 varentropy percentiles ──
# Top 25% by varentropy → hard (keeps ~137 hard math samples)
# 50th–75th percentile  → medium
# Below median          → easy
hard_mask = df["source"].isin(HARD_SOURCES)
if hard_mask.sum() > 0:
    q50 = df.loc[hard_mask, "varentropy"].quantile(0.50)
    q75 = df.loc[hard_mask, "varentropy"].quantile(0.75)
    print(f"Math varentropy thresholds: median={q50:.4f}, q75={q75:.4f}")

    def refine_math(row):
        if row["source"] not in HARD_SOURCES:
            return row["difficulty"]
        if row["varentropy"] >= q75:  return "hard"
        elif row["varentropy"] >= q50: return "medium"
        else:                          return "easy"

    df["difficulty"] = df.apply(refine_math, axis=1)

# ── Step 3: Promote top 15% hardest gsm8k → medium only ──
gsm_mask = df["source"] == "gsm8k"
if gsm_mask.sum() > 0:
    q85_gsm = df.loc[gsm_mask, "varentropy"].quantile(0.85)
    def refine_gsm(row):
        if row["source"] != "gsm8k": return row["difficulty"]
        return "medium" if row["varentropy"] > q85_gsm else "easy"
    df["difficulty"] = df.apply(refine_gsm, axis=1)

# ── Save ──────────────────────────────────────────────────
df.to_csv("data/features/labeled_signals.csv", index=False)

print("\n✅ Final difficulty distribution:")
print(df["difficulty"].value_counts())
print()
print("Distribution by percentage:")
print(df["difficulty"].value_counts(normalize=True).round(3))
print()

counts = df["difficulty"].value_counts(normalize=True)
for cls in ["easy", "medium", "hard"]:
    pct = counts.get(cls, 0)
    icon = "✅" if pct >= 0.18 else "⚠️ "
    print(f"  {icon} '{cls}': {pct:.1%}")

print()
print("Source → Difficulty breakdown:")
print(df.groupby(["source", "difficulty"]).size().to_string())
"""

# scripts/03_label_difficulty.py
# FIX v3: Uses hendrycks_math ground-truth 'level' field
#          instead of varentropy percentiles for math samples.
#
#  Mapping:
#   Level 1-2  → easy    (elementary / early competition)
#   Level 3    → medium  (mid competition)
#   Level 4-5  → hard    (AMC/AIME difficulty)
#
#  Non-math sources keep tier-based labels as before.

import pandas as pd
import json

# ── Load signals flat CSV ─────────────────────────────────
df_signals = pd.read_csv("data/features/signals_flat.csv")

# ── Load the level field from the full dataset JSON ───────
# (signals_flat.csv doesn't have 'level' — we join it back in)
with open("data/processed/unified_dataset.json") as f:
    raw = json.load(f)

level_map = {item["id"]: item.get("level") for item in raw}
df_signals["level"] = df_signals["id"].map(level_map)

print("Sources in signals file:")
print(df_signals["source"].value_counts())
print()

# Show what levels we have for math
math_mask = df_signals["source"].str.startswith("math_")
print("Math level distribution in signals:")
print(df_signals.loc[math_mask, "level"].value_counts().sort_index())
print()

EASY_SOURCES   = {"gsm8k", "mbpp"}
MEDIUM_SOURCES = {"mmlu", "bbh"}
HARD_SOURCES   = {"math_algebra", "math_geometry", "math_number_theory"}

def assign_difficulty(row):
    src = row["source"]

    # ── Math: use ground-truth level ──────────────────────
    if src in HARD_SOURCES:
        lvl = str(row.get("level", "")).strip()
        if lvl in ("Level 1", "Level 2"):  return "easy"
        elif lvl == "Level 3":             return "medium"
        elif lvl in ("Level 4", "Level 5"): return "hard"
        else:
            # Fallback if level missing: use varentropy median split
            return "hard"   # math without level = assume hard

    # ── Easy sources ──────────────────────────────────────
    elif src in EASY_SOURCES:
        return "easy"

    # ── Medium sources ────────────────────────────────────
    elif src in MEDIUM_SOURCES:
        return "medium"

    return "medium"

df_signals["difficulty"] = df_signals.apply(assign_difficulty, axis=1)

# ── Optional: promote top 15% hardest gsm8k → medium ─────
gsm_mask = df_signals["source"] == "gsm8k"
if gsm_mask.sum() > 0:
    q85 = df_signals.loc[gsm_mask, "varentropy"].quantile(0.85)
    def refine_gsm(row):
        if row["source"] != "gsm8k": return row["difficulty"]
        return "medium" if row["varentropy"] > q85 else "easy"
    df_signals["difficulty"] = df_signals.apply(refine_gsm, axis=1)

# ── Save ──────────────────────────────────────────────────
df_signals.to_csv("data/features/labeled_signals.csv", index=False)

print("✅ Final difficulty distribution:")
print(df_signals["difficulty"].value_counts())
print()
print("Distribution by percentage:")
pcts = df_signals["difficulty"].value_counts(normalize=True).round(3)
print(pcts)
print()

for cls in ["easy", "medium", "hard"]:
    pct = pcts.get(cls, 0)
    icon = "✅" if pct >= 0.18 else "⚠️ "
    print(f"  {icon} '{cls}': {pct:.1%}")

print()
print("Source → Difficulty breakdown:")
print(df_signals.groupby(["source","difficulty"]).size().to_string())