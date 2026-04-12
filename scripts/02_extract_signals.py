# scripts/02b_extract_signals_v2.py
# REPLACES 02_extract_signals.py
# 
# KEY CHANGE: Adds 6 math-aware lexical features that actually
# separate hard (proof/advanced math) from medium (word problems).
# These are computed purely with regex + spacy — no LLM needed.
#
# New features:
#   math_symbol_density  — ratio of math symbols (∀∃∈^√∑) per token
#   proof_keyword_count  — "prove","show that","derive","iff","QED" etc.
#   equation_count       — number of equation-like substrings (x^n, f(x))
#   abstract_noun_ratio  — ratio of abstract/formal nouns (theorem, lemma)
#   avg_word_length      — longer words = more technical vocabulary
#   number_density       — ratio of numeric tokens (useful for word problems)

import torch, numpy as np, json, pandas as pd, os, re, spacy
import torch.nn.functional as F
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

MODEL_NAME = "distilgpt2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN    = 256

SAMPLES_PER_SOURCE = {
    "gsm8k":              400,
    "mbpp":               200,
    "math_algebra":       250,
    "math_geometry":      150,
    "math_number_theory": 150,
    "mmlu":               100,
    "bbh":                150,
}

os.makedirs("data/features", exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
if hasattr(model.config, "loss_type") and model.config.loss_type is None:
    model.config.loss_type = "ForCausalLMLoss"
model.eval()

nlp = spacy.load("en_core_web_sm")

with open("data/processed/unified_dataset.json") as f:
    raw_data = json.load(f)

# Stratified sampling
buckets = defaultdict(list)
for item in raw_data:
    buckets[item["source"]].append(item)

data = []
for src, limit in SAMPLES_PER_SOURCE.items():
    pool   = buckets.get(src, [])
    chosen = pool[:limit]
    data.extend(chosen)
    print(f"  {src}: {len(chosen)} samples")

print(f"\nTotal: {len(data)}")

# ── Lexical math features ──────────────────────────────────
PROOF_KEYWORDS = {
    "prove", "proof", "show that", "demonstrate", "derive",
    "theorem", "lemma", "corollary", "iff", "if and only if",
    "qed", "contradiction", "induction", "base case",
    "assume", "suppose", "let", "therefore", "hence",
    "it follows", "we conclude", "by definition"
}
MATH_SYMBOLS_RE = re.compile(r'[∀∃∈∉⊆⊇∩∪∑∏√∞≤≥≠≡±×÷\^]|'
                              r'\\[a-zA-Z]+|'     # LaTeX \alpha etc.
                              r'\d+\^\d+|'        # x^2
                              r'[a-z]\([a-z]\)')  # f(x)

ABSTRACT_NOUNS = {
    "theorem", "lemma", "corollary", "proof", "conjecture",
    "proposition", "axiom", "definition", "hypothesis",
    "equation", "polynomial", "integer", "rational", "irrational",
    "prime", "modulo", "matrix", "vector", "eigenvalue",
    "derivative", "integral", "limit", "convergence"
}

def math_aware_features(text):
    text_lower = text.lower()
    tokens     = text_lower.split()
    n_tokens   = max(len(tokens), 1)

    # 1. Proof keyword count (normalized)
    proof_kw = sum(1 for kw in PROOF_KEYWORDS if kw in text_lower)
    proof_kw_norm = proof_kw / n_tokens

    # 2. Math symbol density
    math_syms  = len(MATH_SYMBOLS_RE.findall(text))
    math_density = math_syms / n_tokens

    # 3. Equation count (patterns like x^2, f(x), x_n, a_ij)
    eq_count = len(re.findall(
        r'[a-zA-Z]\^?\d+|[a-zA-Z]_\{?[a-zA-Z0-9]+\}?|'
        r'[a-zA-Z]\([a-zA-Z0-9,\s]+\)', text))
    eq_density = eq_count / n_tokens

    # 4. Abstract noun ratio
    abs_nouns  = sum(1 for t in tokens if t.strip('.,;:') in ABSTRACT_NOUNS)
    abs_ratio  = abs_nouns / n_tokens

    # 5. Average word length (technical text has longer words)
    clean_tokens = [t.strip('.,;:!?()[]{}') for t in tokens if t.strip('.,;:!?()[]{}')]
    avg_word_len = float(np.mean([len(t) for t in clean_tokens])) if clean_tokens else 0.0

    # 6. Number density (word problems have lots of numbers)
    num_tokens   = sum(1 for t in tokens if re.match(r'^\d+\.?\d*$', t))
    num_density  = num_tokens / n_tokens

    return proof_kw_norm, math_density, eq_density, abs_ratio, avg_word_len, num_density


# ── Neural signals (same as before) ───────────────────────
def extract_neural_signals(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN, truncation=True,
                       padding=True).to(DEVICE)
    with torch.no_grad():
        out = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
        )

    hidden    = out.hidden_states[-1]
    embedding = hidden.mean(dim=1).squeeze().cpu().numpy()

    attn_entropy = 0.0
    if out.attentions:
        ents = []
        for la in out.attentions:
            p = la.squeeze(0) + 1e-9
            H = -torch.sum(p * torch.log(p), dim=-1)
            ents.append(H.mean().item())
        attn_entropy = float(np.mean(ents))

    logits     = out.logits.squeeze(0)
    probs      = torch.softmax(logits, dim=-1) + 1e-9
    entropy    = -torch.sum(probs * torch.log(probs), dim=-1)
    varentropy = float(entropy.var().item())

    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
    )
    perplexity = float(torch.exp(ce).item())

    return embedding, attn_entropy, varentropy, perplexity


# ── Syntax features ───────────────────────────────────────
def syntax_features(text):
    doc   = nlp(text[:1000])
    sents = list(doc.sents)
    if not sents:
        return 0.0, 0, 0.0
    max_depth = 0
    for token in doc:
        d, t = 0, token
        while t.head != t:
            t = t.head; d += 1
        max_depth = max(max_depth, d)
    clause_count = sum(1 for t in doc
                       if t.dep_ in ("relcl","advcl","ccomp","xcomp"))
    avg_slen = float(np.mean([len(s) for s in sents]))
    return float(max_depth), clause_count, avg_slen


# ── Main loop ─────────────────────────────────────────────
records = []

for item in tqdm(data, desc="Extracting signals v2"):
    q = item["question"]
    try:
        emb, attn_ent, var_ent, ppl = extract_neural_signals(q)
        tree_depth, clause_cnt, avg_slen = syntax_features(q)
        (proof_kw, math_sym, eq_dens,
         abs_ratio, avg_wlen, num_dens) = math_aware_features(q)

        records.append({
            "id":               item["id"],
            "source":           item["source"],
            "question":         q,
            # original neural signals
            "attn_entropy":     attn_ent,
            "varentropy":       var_ent,
            "perplexity":       ppl,
            # syntax signals
            "tree_depth":       tree_depth,
            "clause_count":     clause_cnt,
            "avg_sent_len":     avg_slen,
            # NEW math-aware signals
            "proof_kw_density": proof_kw,
            "math_sym_density": math_sym,
            "eq_density":       eq_dens,
            "abstract_ratio":   abs_ratio,
            "avg_word_len":     avg_wlen,
            "num_density":      num_dens,
            # embedding
            "embedding":        emb.tolist(),
            "raw_difficulty":   item["raw_difficulty"]
        })
    except Exception as e:
        print(f"Skipped {item['id']}: {e}")

df = pd.DataFrame(records)
df.to_json("data/features/signals_v2.json", orient="records", indent=2)
df.drop(columns=["embedding"]).to_csv(
    "data/features/signals_v2_flat.csv", index=False)

print(f"\n✅ Saved {len(df)} records to data/features/signals_v2.json")
print("\nSample math-aware feature means per source:")
print(df.groupby("source")[
    ["proof_kw_density","math_sym_density","eq_density","abstract_ratio"]
].mean().round(4).to_string())
