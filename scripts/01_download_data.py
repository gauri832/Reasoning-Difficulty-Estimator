# scripts/01_download_data.py
# FIX v3: Preserves the 'level' field from hendrycks_math
#          so labeler can use ground-truth difficulty (Level 1-5).

from datasets import load_dataset
import json, os, random

os.makedirs("data/processed", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

datasets_to_load = {
    "gsm8k":              ("gsm8k",                     "main"),
    "math_algebra":       ("EleutherAI/hendrycks_math",  "algebra"),
    "math_geometry":      ("EleutherAI/hendrycks_math",  "geometry"),
    "math_number_theory": ("EleutherAI/hendrycks_math",  "number_theory"),
    "mmlu":               ("cais/mmlu",                  "abstract_algebra"),
    "bbh":                ("lukaemon/bbh",               "causal_judgement"),
    "mbpp":               ("mbpp",                       "sanitized"),
}

unified = []

for name, (hf_path, subset) in datasets_to_load.items():
    try:
        ds = load_dataset(hf_path, subset) if subset else load_dataset(hf_path)

        if "test" in ds:      ds = ds["test"]
        elif "validation" in ds: ds = ds["validation"]
        else:                 ds = ds["train"]

        count = 0
        for item in ds:
            question = (
                item.get("question") or item.get("problem") or
                item.get("input")    or item.get("prompt")  or ""
            )
            answer = (
                item.get("answer")   or item.get("solution") or
                item.get("target")   or ""
            )

            # ✅ KEY FIX: preserve 'level' for hendrycks_math samples
            level = item.get("level", None)   # e.g. "Level 1" ... "Level 5"

            unified.append({
                "id":             f"{name}_{len(unified)}",
                "source":         name,
                "question":       question,
                "answer":         answer,
                "level":          level,       # NEW field
                "raw_difficulty": None
            })
            count += 1
        print(f"✅ {name}: {count} samples loaded")

    except Exception as e:
        print(f"❌ Failed {name}: {e}")

random.seed(42)
random.shuffle(unified)

with open("data/processed/unified_dataset.json", "w") as f:
    json.dump(unified, f, indent=2)

print(f"\n✅ Total samples: {len(unified)} (shuffled)")

# Quick check: show level distribution for math
math_items = [x for x in unified if x["source"].startswith("math_")]
from collections import Counter
levels = Counter(x["level"] for x in math_items)
print("\nMath level distribution:")
for lvl in sorted(levels):
    print(f"  {lvl}: {levels[lvl]}")