# scripts/02_extract_signals.py
"""
import torch
import numpy as np
import json
import pandas as pd
import spacy
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "meta-llama/Llama-3.2-1B"   # Small encoder for speed
# Alternatively: "mistralai/Mistral-7B-v0.1" for richer signals
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, output_attentions=True).to(DEVICE)
model.eval()
nlp = spacy.load("en_core_web_sm")

with open("data/processed/unified_dataset.json") as f:
    data = json.load(f)

# ── Signal 1: Prompt Embedding ───────────────────────────
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN, truncation=True,
                       padding=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    # CLS token from last hidden state
    return out.last_hidden_state[:, 0, :].squeeze().cpu().numpy()

# ── Signal 2: Attention Entropy ──────────────────────────
def attention_entropy(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    # out.attentions: tuple of (batch, heads, seq, seq)
    # Average across all layers and heads
    entropies = []
    for layer_attn in out.attentions:
        # layer_attn: (1, num_heads, seq_len, seq_len)
        p = layer_attn.squeeze(0)           # (heads, seq, seq)
        p = p + 1e-9                         # numerical stability
        H = -torch.sum(p * torch.log(p), dim=-1)   # (heads, seq)
        entropies.append(H.mean().item())
    return float(np.mean(entropies))

# ── Signal 3: Token Varentropy ───────────────────────────
def token_varentropy(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN, truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model(**inputs)
    hidden = out.last_hidden_state.squeeze(0)   # (seq_len, hidden)
    # Project to vocab logits and compute per-token entropy
    logits = model.lm_head(hidden) if hasattr(model, 'lm_head') else hidden @ model.embed_tokens.weight.T
    probs = torch.softmax(logits, dim=-1) + 1e-9
    token_entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # (seq_len,)
    return float(token_entropy.var().item())

# ── Signal 4: Perplexity ─────────────────────────────────
def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN, truncation=True).to(DEVICE)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        out = model(**inputs, labels=input_ids)
    # out.loss = mean NLL
    return float(torch.exp(out.loss).item()) if hasattr(out, 'loss') and out.loss else 0.0

# ── Signal 5: Syntax Complexity ──────────────────────────
def syntax_complexity(text):
    doc = nlp(text[:1000])          # cap for speed
    sentences = list(doc.sents)
    if not sentences:
        return 0.0, 0, 0.0
    max_depth = 0
    for token in doc:
        depth = 0
        t = token
        while t.head != t:
            t = t.head
            depth += 1
        max_depth = max(max_depth, depth)
    clause_count = sum(1 for t in doc if t.dep_ in ("relcl","advcl","ccomp","xcomp"))
    avg_sent_len = np.mean([len(s) for s in sentences])
    return float(max_depth), int(clause_count), float(avg_sent_len)

# ── Main Extraction Loop ──────────────────────────────────
records = []
for item in tqdm(data, desc="Extracting signals"):
    q = item["question"]
    try:
        emb = get_embedding(q)
        attn_ent = attention_entropy(q)
        var_ent = token_varentropy(q)
        ppl = compute_perplexity(q)
        tree_depth, clause_cnt, avg_slen = syntax_complexity(q)

        records.append({
            "id": item["id"],
            "source": item["source"],
            "question": q,
            "attn_entropy": attn_ent,
            "varentropy": var_ent,
            "perplexity": ppl,
            "tree_depth": tree_depth,
            "clause_count": clause_cnt,
            "avg_sent_len": avg_slen,
            "embedding": emb.tolist(),    # stored as list
            "raw_difficulty": item["raw_difficulty"]
        })
    except Exception as e:
        print(f"Skipped {item['id']}: {e}")

df = pd.DataFrame(records)
df.to_json("data/features/signals.json", orient="records", indent=2)
# Also save a flat CSV (without embedding) for inspection
df.drop(columns=["embedding"]).to_csv("data/features/signals_flat.csv", index=False)
print(f"Saved {len(df)} signal records.")



# scripts/02_extract_signals.py

import torch
import numpy as np
import json
import pandas as pd
import os
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "distilgpt2"   # ✅ Open model (no access issues)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256

# Create folders
os.makedirs("data/features", exist_ok=True)

# Load model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_attentions=True
).to(DEVICE)

model.eval()
nlp = spacy.load("en_core_web_sm")

# Load data
with open("data/processed/unified_dataset.json") as f:
    data = json.load(f)

# ── Signal 1: Embedding (mean pooled) ────────────────────
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN,
                       truncation=True,
                       padding=True).to(DEVICE)

    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)

    hidden = out.hidden_states[-1]  # last layer
    return hidden.mean(dim=1).squeeze().cpu().numpy()

# ── Signal 2: Attention Entropy ──────────────────────────
def attention_entropy(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN,
                       truncation=True).to(DEVICE)

    with torch.no_grad():
        out = model(**inputs)

    entropies = []
    for layer_attn in out.attentions:
        p = layer_attn.squeeze(0)
        p = p + 1e-9
        H = -torch.sum(p * torch.log(p), dim=-1)
        entropies.append(H.mean().item())

    return float(np.mean(entropies))

# ── Signal 3: Token Varentropy ───────────────────────────
def token_varentropy(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN,
                       truncation=True).to(DEVICE)

    with torch.no_grad():
        out = model(**inputs)

    logits = out.logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1) + 1e-9
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)

    return float(entropy.var().item())

# ── Signal 4: Perplexity ─────────────────────────────────
def compute_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN,
                       truncation=True).to(DEVICE)

    input_ids = inputs["input_ids"]

    with torch.no_grad():
        out = model(**inputs, labels=input_ids)

    return float(torch.exp(out.loss).item())

# ── Signal 5: Syntax Complexity ──────────────────────────
def syntax_complexity(text):
    doc = nlp(text[:1000])

    sentences = list(doc.sents)
    if not sentences:
        return 0.0, 0, 0.0

    max_depth = 0
    for token in doc:
        depth = 0
        t = token
        while t.head != t:
            t = t.head
            depth += 1
        max_depth = max(max_depth, depth)

    clause_count = sum(1 for t in doc if t.dep_ in ("relcl","advcl","ccomp","xcomp"))
    avg_sent_len = np.mean([len(s) for s in sentences])

    return float(max_depth), int(clause_count), float(avg_sent_len)

# ── Main Loop ────────────────────────────────────────────
records = []

for item in tqdm(data[:2000], desc="Extracting signals"):  # 🔥 limit for speed
    q = item["question"]

    try:
        emb = get_embedding(q)
        attn_ent = attention_entropy(q)
        var_ent = token_varentropy(q)
        ppl = compute_perplexity(q)
        tree_depth, clause_cnt, avg_slen = syntax_complexity(q)

        records.append({
            "id": item["id"],
            "source": item["source"],
            "question": q,
            "attn_entropy": attn_ent,
            "varentropy": var_ent,
            "perplexity": ppl,
            "tree_depth": tree_depth,
            "clause_count": clause_cnt,
            "avg_sent_len": avg_slen,
            "embedding": emb.tolist(),
            "raw_difficulty": item["raw_difficulty"]
        })

    except Exception as e:
        print(f"Skipped {item['id']}: {e}")

# Save outputs
df = pd.DataFrame(records)

df.to_json("data/features/signals.json", orient="records", indent=2)
df.drop(columns=["embedding"]).to_csv("data/features/signals_flat.csv", index=False)

print(f"Saved {len(df)} signal records.")


# scripts/02_extract_signals.py

import torch
import numpy as np
import json
import pandas as pd
import os
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── Config ──────────────────────────────────────────────
MODEL_NAME = "distilgpt2"   # ✅ open + fast
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256
LIMIT = 1000   # 🔥 reduce for speed (increase later)

# Create folders
os.makedirs("data/features", exist_ok=True)

# Load tokenizer + fix padding issue ✅
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_attentions=True
).to(DEVICE)

model.eval()

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Load dataset
with open("data/processed/unified_dataset.json") as f:
    data = json.load(f)

# ── Optimized Signal Extraction (Single Forward Pass) ────
def extract_all_signals(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        out = model(
            **inputs,
            labels=inputs["input_ids"],
            output_hidden_states=True
        )

    # ── Embedding (mean pooling) ──
    hidden = out.hidden_states[-1]
    embedding = hidden.mean(dim=1).squeeze().cpu().numpy()

    # ── Attention Entropy ──
    attn_entropy = 0.0
    if out.attentions:
        entropies = []
        for layer_attn in out.attentions:
            p = layer_attn.squeeze(0) + 1e-9
            H = -torch.sum(p * torch.log(p), dim=-1)
            entropies.append(H.mean().item())
        attn_entropy = float(np.mean(entropies))

    # ── Token Varentropy ──
    logits = out.logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1) + 1e-9
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    varentropy = float(entropy.var().item())

    # ── Perplexity ──
    perplexity = float(torch.exp(out.loss).item())

    return embedding, attn_entropy, varentropy, perplexity


# ── Syntax Complexity ────────────────────────────────────
def syntax_complexity(text):
    doc = nlp(text[:1000])

    sentences = list(doc.sents)
    if not sentences:
        return 0.0, 0, 0.0

    max_depth = 0
    for token in doc:
        depth = 0
        t = token
        while t.head != t:
            t = t.head
            depth += 1
        max_depth = max(max_depth, depth)

    clause_count = sum(
        1 for t in doc if t.dep_ in ("relcl", "advcl", "ccomp", "xcomp")
    )

    avg_sent_len = np.mean([len(s) for s in sentences])

    return float(max_depth), int(clause_count), float(avg_sent_len)


# ── Main Loop ────────────────────────────────────────────
records = []

for item in tqdm(data[:LIMIT], desc="Extracting signals"):
    q = item["question"]

    try:
        emb, attn_ent, var_ent, ppl = extract_all_signals(q)
        tree_depth, clause_cnt, avg_slen = syntax_complexity(q)

        records.append({
            "id": item["id"],
            "source": item["source"],
            "question": q,
            "attn_entropy": attn_ent,
            "varentropy": var_ent,
            "perplexity": ppl,
            "tree_depth": tree_depth,
            "clause_count": clause_cnt,
            "avg_sent_len": avg_slen,
            "embedding": emb.tolist(),
            "raw_difficulty": item["raw_difficulty"]
        })

    except Exception as e:
        print(f"Skipped {item['id']}: {e}")

# ── Save Results ─────────────────────────────────────────
df = pd.DataFrame(records)

df.to_json("data/features/signals.json", orient="records", indent=2)
df.drop(columns=["embedding"]).to_csv(
    "data/features/signals_flat.csv", index=False
)

print(f"✅ Saved {len(df)} signal records.")
"""
# scripts/02_extract_signals.py
# FIX: Uses stratified sampling so all sources are represented
#      in the extracted signals (not just gsm8k top-1000).

import torch
import numpy as np
import json
import pandas as pd
import os
import spacy
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────
MODEL_NAME  = "distilgpt2"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN     = 256

# CRITICAL FIX: sample proportionally per source, not just top-N
SAMPLES_PER_SOURCE = {
    "gsm8k":              400,   # easy
    "mbpp":               200,   # easy
    "math_algebra":       250,   # hard
    "math_geometry":      150,   # hard
    "math_number_theory": 150,   # hard
    "mmlu":               100,   # medium
    "bbh":                150,   # medium
}
# Total = ~1400 samples, balanced across difficulty tiers

os.makedirs("data/features", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, output_attentions=True
).to(DEVICE)
model.eval()

nlp = spacy.load("en_core_web_sm")

with open("data/processed/unified_dataset.json") as f:
    raw_data = json.load(f)

# ── Stratified sampling ───────────────────────────────────
buckets = defaultdict(list)
for item in raw_data:
    buckets[item["source"]].append(item)

data = []
for src, limit in SAMPLES_PER_SOURCE.items():
    pool = buckets.get(src, [])
    chosen = pool[:limit]
    data.extend(chosen)
    print(f"  {src}: {len(chosen)} samples selected (pool: {len(pool)})")

print(f"\nTotal for extraction: {len(data)}")

# ── Single forward pass extracts all 4 neural signals ─────
def extract_all_signals(text):
    inputs = tokenizer(
        text, return_tensors="pt",
        max_length=MAX_LEN, truncation=True, padding=True
    ).to(DEVICE)

    with torch.no_grad():
        out = model(
            **inputs,
            labels=inputs["input_ids"],
            output_hidden_states=True
        )

    # Signal 1: Embedding (mean pooling of last hidden layer)
    hidden    = out.hidden_states[-1]
    embedding = hidden.mean(dim=1).squeeze().cpu().numpy()

    # Signal 2: Attention Entropy
    attn_entropy = 0.0
    if out.attentions:
        entropies = []
        for layer_attn in out.attentions:
            p = layer_attn.squeeze(0) + 1e-9
            H = -torch.sum(p * torch.log(p), dim=-1)
            entropies.append(H.mean().item())
        attn_entropy = float(np.mean(entropies))

    # Signal 3: Token Varentropy
    logits  = out.logits.squeeze(0)
    probs   = torch.softmax(logits, dim=-1) + 1e-9
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    varentropy = float(entropy.var().item())

    # Signal 4: Perplexity
    perplexity = float(torch.exp(out.loss).item())

    return embedding, attn_entropy, varentropy, perplexity


# ── Signal 5: Syntax complexity (no model needed) ─────────
def syntax_complexity(text):
    doc       = nlp(text[:1000])
    sentences = list(doc.sents)
    if not sentences:
        return 0.0, 0, 0.0

    max_depth = 0
    for token in doc:
        depth, t = 0, token
        while t.head != t:
            t = t.head
            depth += 1
        max_depth = max(max_depth, depth)

    clause_count = sum(
        1 for t in doc if t.dep_ in ("relcl", "advcl", "ccomp", "xcomp")
    )
    avg_sent_len = float(np.mean([len(s) for s in sentences]))

    return float(max_depth), int(clause_count), avg_sent_len


# ── Main extraction loop ───────────────────────────────────
records = []

for item in tqdm(data, desc="Extracting signals"):
    q = item["question"]
    try:
        emb, attn_ent, var_ent, ppl = extract_all_signals(q)
        tree_depth, clause_cnt, avg_slen = syntax_complexity(q)

        records.append({
            "id":           item["id"],
            "source":       item["source"],
            "question":     q,
            "attn_entropy": attn_ent,
            "varentropy":   var_ent,
            "perplexity":   ppl,
            "tree_depth":   tree_depth,
            "clause_count": clause_cnt,
            "avg_sent_len": avg_slen,
            "embedding":    emb.tolist(),
            "raw_difficulty": item["raw_difficulty"]
        })
    except Exception as e:
        print(f"Skipped {item['id']}: {e}")

df = pd.DataFrame(records)
df.to_json("data/features/signals.json", orient="records", indent=2)
df.drop(columns=["embedding"]).to_csv("data/features/signals_flat.csv", index=False)

print(f"\n✅ Saved {len(df)} signal records.")
print("\nSource distribution in extracted signals:")
print(df["source"].value_counts())