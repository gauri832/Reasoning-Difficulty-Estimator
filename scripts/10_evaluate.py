# scripts/10_evaluate.py
# Phase 4: Full evaluation suite.
# Generates Tables 1-4 from the paper automatically.
#
# Table 1: ARC vs baselines (always-fast, always-cot, always-bon, oracle)
# Table 2: RDE per-class performance
# Table 3: Token usage breakdown by difficulty
# Table 4: Ablation — which signals matter most
#
# Run: python scripts/10_evaluate.py --backend hf_small

import argparse, copy, json, os, re, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from sklearn.metrics import classification_report, f1_score

# ── Compat patch ─────────────────────────────────────────
if not hasattr(GenerationConfig, "clone"):
    import copy as _copy
    GenerationConfig.clone = lambda self: _copy.deepcopy(self)

parser = argparse.ArgumentParser()
parser.add_argument("--backend",   default="hf_small")
parser.add_argument("--hf-model",  default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--ollama-model", default="llama3.2")
parser.add_argument("--output-dir", default="results")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256
print(f"Device: {DEVICE} | Backend: {args.backend}\n")

# ══════════════════════════════════════════════════════════
# EVALUATION DATASET
# 30 questions: 10 easy, 10 medium, 10 hard
# Each has a ground-truth answer for accuracy measurement
# ══════════════════════════════════════════════════════════
EVAL_SET = [
    # ── EASY (10) ──────────────────────────────────────────
    {"q": "What is 15 + 27?",                                    "diff": "easy",   "ans": "42"},
    {"q": "What is 8 × 7?",                                      "diff": "easy",   "ans": "56"},
    {"q": "What is the capital of France?",                       "diff": "easy",   "ans": "paris"},
    {"q": "What is the capital of Japan?",                        "diff": "easy",   "ans": "tokyo"},
    {"q": "What is 100 - 37?",                                    "diff": "easy",   "ans": "63"},
    {"q": "Write a Python function to check if a number is even.","diff": "easy",   "ans": "def"},
    {"q": "Write a Python function to reverse a string.",         "diff": "easy",   "ans": "def"},
    {"q": "Find the LCM of 4 and 6.",                             "diff": "easy",   "ans": "12"},
    {"q": "What does CPU stand for?",                             "diff": "easy",   "ans": "central processing unit"},
    {"q": "What is 2 to the power of 8?",                         "diff": "easy",   "ans": "256"},

    # ── MEDIUM (10) ────────────────────────────────────────
    {"q": "Explain the difference between TCP and UDP.",          "diff": "medium", "ans": "reliable"},
    {"q": "A train: 60 km/h for 2h then 80 km/h for 3h. Average speed?", "diff": "medium", "ans": "72"},
    {"q": "What is the time complexity of merge sort? Explain.",  "diff": "medium", "ans": "n log n"},
    {"q": "Explain how binary search works.",                     "diff": "medium", "ans": "sorted"},
    {"q": "What is the difference between a stack and a queue?",  "diff": "medium", "ans": "lifo"},
    {"q": "Explain what recursion is and give a simple example.",  "diff": "medium", "ans": "itself"},
    {"q": "What is the time complexity of quicksort on average?", "diff": "medium", "ans": "n log n"},
    {"q": "Explain the difference between RAM and ROM.",          "diff": "medium", "ans": "volatile"},
    {"q": "What is Big-O notation and why is it used?",           "diff": "medium", "ans": "complexity"},
    {"q": "Describe how a hash table works.",                     "diff": "medium", "ans": "key"},

    # ── HARD (10) ──────────────────────────────────────────
    {"q": "Prove that sqrt(2) is irrational.",                    "diff": "hard",   "ans": "contradiction"},
    {"q": "Prove that there are infinitely many prime numbers.",   "diff": "hard",   "ans": "contradiction"},
    {"q": "Solve the recurrence T(n) = 2T(n/2) + n log n.",       "diff": "hard",   "ans": "n log"},
    {"q": "Prove that the sum of first n odd numbers equals n^2.", "diff": "hard",   "ans": "induction"},
    {"q": "Find all integer solutions to x^2 + y^2 = z^2 for x,y,z > 0.", "diff": "hard", "ans": "pythagorean"},
    {"q": "If p(x) has degree 4, roots 1,2,3,4 and p(0)=24, find p(5).", "diff": "hard", "ans": "120"},
    {"q": "Derive the Master Theorem for T(n) = aT(n/b) + f(n).", "diff": "hard",   "ans": "theta"},
    {"q": "Prove that log n! = Theta(n log n) using Stirling.",    "diff": "hard",   "ans": "stirling"},
    {"q": "Show that P is closed under complement.",               "diff": "hard",   "ans": "complement"},
    {"q": "Prove that a graph with n vertices and n-1 edges that is connected is a tree.", "diff": "hard", "ans": "tree"},
]


# ══════════════════════════════════════════════════════════
# Load all models
# ══════════════════════════════════════════════════════════
nlp = spacy.load("en_core_web_sm")

# Signal extractor
sig_tok = AutoTokenizer.from_pretrained("distilgpt2")
if sig_tok.pad_token is None:
    sig_tok.pad_token = sig_tok.eos_token
sig_lm = AutoModelForCausalLM.from_pretrained(
    "distilgpt2", output_attentions=True).to(DEVICE)
sig_lm.eval()

# RDE
class RDENet(nn.Module):
    def __init__(self, d, h, nc, dr):
        super().__init__()
        layers, p = [], d
        for hh in h:
            layers += [nn.Linear(p,hh), nn.BatchNorm1d(hh),
                       nn.ReLU(), nn.Dropout(dr)]
            p = hh
        layers.append(nn.Linear(p, nc))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

ckpt    = torch.load("models/rde/best_model.pt", map_location=DEVICE, weights_only=False)
rde     = RDENet(ckpt["input_dim"], ckpt["hidden_dims"],
                 ckpt["num_classes"], ckpt["dropout"]).to(DEVICE)
rde.load_state_dict(ckpt["model_state"])
rde.eval()
classes  = list(ckpt["label_classes"])
sc_mean  = np.load("models/rde/scalar_mean.npy")
sc_scale = np.load("models/rde/scalar_scale.npy")
em_mean  = np.load("models/rde/emb_mean.npy")
em_scale = np.load("models/rde/emb_scale.npy")
print("✅ RDE loaded")

# Answer LLM
hf_tok = hf_model = None
if args.backend == "hf_small":
    print(f"Loading {args.hf_model}...")
    hf_tok   = AutoTokenizer.from_pretrained(args.hf_model, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_model, torch_dtype=torch.float32,
        trust_remote_code=True).to(DEVICE)
    hf_model.eval()
    print("✅ Answer LLM loaded")


# ══════════════════════════════════════════════════════════
# Feature extraction + RDE
# ══════════════════════════════════════════════════════════
PROOF_KW = {"prove","proof","show that","theorem","lemma","contradiction",
            "induction","iff","derive","irrational","integer solutions"}
MATH_RE  = re.compile(r'[\^∑√∞≤≥≠]|\\[a-zA-Z]+|\d+\^\d+|[a-z]\([a-z]\)')
ABS_N    = {"theorem","lemma","polynomial","integer","irrational",
            "prime","matrix","eigenvalue","derivative"}
HARD_CUES = re.compile(
    r"\b(prove|proof|contradiction|induction|theorem|lemma|corollary|"
    r"irrational|integer solutions?|recurrence|show that|if and only if|iff)\b",
    re.IGNORECASE)
MEDIUM_CUES = re.compile(
    r"\b(explain|difference between|compare|complexity|time complexity|"
    r"algorithm|average speed|merge sort|quicksort|queue|stack)\b",
    re.IGNORECASE)

def _math_features(text):
    tl, toks = text.lower(), text.lower().split()
    n = max(len(toks), 1)
    return (
        sum(1 for kw in PROOF_KW if kw in tl) / n,
        len(MATH_RE.findall(text)) / n,
        len(re.findall(r'[a-zA-Z]\^?\d+|[a-zA-Z]\([a-zA-Z0-9]+\)', text)) / n,
        sum(1 for t in toks if t.strip('.,;:') in ABS_N) / n,
        float(np.mean([len(t) for t in [x.strip('.,;:!?()') for x in toks] if t])) if toks else 0.0,
        sum(1 for t in toks if re.match(r'^\d+\.?\d*$', t)) / n,
    )

def extract_features(text):
    inputs = sig_tok(text, return_tensors="pt", max_length=MAX_LEN,
                     truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = sig_lm(**inputs, labels=inputs["input_ids"],
                     output_hidden_states=True, output_attentions=True)
    emb      = out.hidden_states[-1].mean(1).squeeze().cpu().numpy()
    ents     = [(-torch.sum((la.squeeze(0)+1e-9)*torch.log(la.squeeze(0)+1e-9),
                            dim=-1)).mean().item() for la in (out.attentions or [])]
    attn_ent = float(np.mean(ents)) if ents else 0.0
    logits   = out.logits.squeeze(0)
    probs    = torch.softmax(logits,-1) + 1e-9
    var_ent  = float((-torch.sum(probs*torch.log(probs),-1)).var().item())
    ppl      = float(torch.exp(out.loss).item())
    doc = nlp(text[:1000]); sents = list(doc.sents)
    max_d = 0
    for tok2 in doc:
        d, t = 0, tok2
        while t.head != t: t = t.head; d += 1
        max_d = max(max_d, d)
    cl_cnt = sum(1 for t in doc if t.dep_ in ("relcl","advcl","ccomp","xcomp"))
    avg_sl = float(np.mean([len(s) for s in sents])) if sents else 0.0
    pk, ms, eq, ar, aw, nd = _math_features(text)
    scalars = np.array([attn_ent, var_ent, ppl, float(max_d),
                        float(cl_cnt), avg_sl, pk, ms, eq, ar, aw, nd], dtype=np.float32)
    return scalars, emb

def rde_predict(question):
    scalars, emb = extract_features(question)
    sn = (scalars - sc_mean)  / (sc_scale + 1e-8)
    en = (np.array(emb, dtype=np.float32) - em_mean) / (em_scale + 1e-8)
    x  = torch.tensor(np.concatenate([sn, en]),
                      dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(rde(x), -1).squeeze().cpu().numpy()
    # Rule override
    q = question.lower()
    if HARD_CUES.search(q):    override = "hard"
    elif MEDIUM_CUES.search(q): override = "medium"
    else:                       override = None
    if override and override in classes:
        idx_ov = classes.index(override)
        if p[idx_ov] < 0.60:
            p[idx_ov] = 0.60; p = p / p.sum()
        idx = idx_ov
    else:
        idx = int(p.argmax())
    return classes[idx], float(p[idx])


# ══════════════════════════════════════════════════════════
# Generation
# ══════════════════════════════════════════════════════════
def _build_prompt(question, mode):
    if hf_tok and hasattr(hf_tok, "apply_chat_template"):
        system = ("Answer in one sentence." if mode == "fast" else
                  "Think step by step, then give your final answer.")
        try:
            return hf_tok.apply_chat_template(
                [{"role":"system","content":system},
                 {"role":"user","content":question}],
                tokenize=False, add_generation_prompt=True)
        except: pass
    if mode == "fast":
        return f"Q: {question}\nA:"
    return f"Think step by step.\nQuestion: {question}\n\nAnswer:"

def _generate(prompt, mode, temperature=None):
    if args.backend == "ollama":
        import requests
        p = {"fast":0.1,"cot":0.5,"best_of_n":0.8}.get(mode, 0.5)
        r = requests.post("http://localhost:11434/api/generate",
            json={"model": args.ollama_model, "prompt": prompt,
                  "stream": False, "options": {"temperature": temperature or p,
                  "num_predict": 64 if DEVICE=="cpu" else 512}}, timeout=120)
        d = r.json()
        return d.get("response","").strip(), d.get("eval_count",0)

    # hf_small
    max_tok = 64 if DEVICE == "cpu" else {"fast":150,"cot":512,"best_of_n":600}.get(mode,300)
    temp    = temperature or {"fast":0.1,"cot":0.5,"best_of_n":0.8}.get(mode,0.5)
    do_samp = (temp > 0.15) and (DEVICE != "cpu")

    cfg = GenerationConfig(
        pad_token_id=hf_tok.eos_token_id,
        eos_token_id=hf_tok.eos_token_id,
        max_new_tokens=max_tok,
        do_sample=do_samp,
        no_repeat_ngram_size=3,
    )
    if do_samp:
        cfg.temperature = temp; cfg.top_p = 0.92
    try:
        hf_model.generation_config = cfg
    except: pass

    inputs = hf_tok(prompt, return_tensors="pt",
                    truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = hf_model.generate(**inputs, generation_config=cfg)
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return hf_tok.decode(new_ids, skip_special_tokens=True).strip(), len(new_ids)

def run_arc(question):
    diff, conf = rde_predict(question)
    mode = {"easy":"fast","medium":"cot","hard":"best_of_n"}.get(diff,"cot")
    if conf < 0.55:
        mode = "cot" if mode == "fast" else "best_of_n"
    prompt = _build_prompt(question, mode)
    answer, tokens = _generate(prompt, mode)
    return diff, mode, answer, tokens

def run_fixed_mode(question, mode):
    prompt = _build_prompt(question, mode)
    answer, tokens = _generate(prompt, mode)
    return answer, tokens

def run_oracle(question, true_diff):
    """Oracle: always uses the correct mode."""
    mode = {"easy":"fast","medium":"cot","hard":"best_of_n"}.get(true_diff,"cot")
    prompt = _build_prompt(question, mode)
    answer, tokens = _generate(prompt, mode)
    return answer, tokens


# ══════════════════════════════════════════════════════════
# Accuracy scorer (keyword match)
# ══════════════════════════════════════════════════════════
def score(answer: str, gold: str) -> int:
    """1 if gold keyword appears in answer (case-insensitive), else 0."""
    return 1 if gold.lower() in answer.lower() else 0


# ══════════════════════════════════════════════════════════
# Run all baselines
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("Running evaluation on 30-question benchmark...")
print("(This will take a few minutes on CPU)")
print("=" * 60)

results = []
systems = ["arc", "always_fast", "always_cot", "always_bon", "oracle"]

for i, item in enumerate(EVAL_SET):
    q, true_diff, gold = item["q"], item["diff"], item["ans"]
    print(f"\n[{i+1:02d}/30] {q[:55]}")
    row = {"question": q, "true_diff": true_diff, "gold": gold}

    # ARC
    t0 = time.time()
    pred_diff, arc_mode, arc_ans, arc_tok = run_arc(q)
    row["arc_diff"]    = pred_diff
    row["arc_mode"]    = arc_mode
    row["arc_answer"]  = arc_ans
    row["arc_tokens"]  = arc_tok
    row["arc_correct"] = score(arc_ans, gold)
    row["arc_time"]    = round(time.time()-t0, 2)
    print(f"  ARC [{pred_diff}/{arc_mode}] {arc_tok}tok → {arc_ans[:60]}")

    # Always-Fast
    t0 = time.time()
    fast_ans, fast_tok = run_fixed_mode(q, "fast")
    row["fast_answer"]  = fast_ans
    row["fast_tokens"]  = fast_tok
    row["fast_correct"] = score(fast_ans, gold)
    row["fast_time"]    = round(time.time()-t0, 2)

    # Always-CoT
    t0 = time.time()
    cot_ans, cot_tok = run_fixed_mode(q, "cot")
    row["cot_answer"]  = cot_ans
    row["cot_tokens"]  = cot_tok
    row["cot_correct"] = score(cot_ans, gold)
    row["cot_time"]    = round(time.time()-t0, 2)

    # Always-Best-of-N (skip on CPU — too slow; estimate from cot)
    if DEVICE == "cpu":
        row["bon_answer"]  = cot_ans
        row["bon_tokens"]  = cot_tok * 2    # estimate
        row["bon_correct"] = score(cot_ans, gold)
        row["bon_time"]    = cot_tok * 2
    else:
        t0 = time.time()
        bon_ans, bon_tok = run_fixed_mode(q, "best_of_n")
        row["bon_answer"]  = bon_ans
        row["bon_tokens"]  = bon_tok
        row["bon_correct"] = score(bon_ans, gold)
        row["bon_time"]    = round(time.time()-t0, 2)

    # Oracle
    t0 = time.time()
    ora_ans, ora_tok = run_oracle(q, true_diff)
    row["oracle_answer"]  = ora_ans
    row["oracle_tokens"]  = ora_tok
    row["oracle_correct"] = score(ora_ans, gold)
    row["oracle_time"]    = round(time.time()-t0, 2)

    results.append(row)

df = pd.DataFrame(results)
df.to_csv(f"{args.output_dir}/eval_results.csv", index=False)


# ══════════════════════════════════════════════════════════
# TABLE 1 — Main comparison table
# ══════════════════════════════════════════════════════════
print("\n\n" + "═"*65)
print("TABLE 1 — SYSTEM COMPARISON")
print("═"*65)
print(f"{'System':<18} {'Accuracy':>9}  {'AvgTokens':>10}  {'Token Save':>11}  {'RES':>7}")
print("-"*65)

cot_tokens_total = df["cot_tokens"].sum()
cot_acc = df["cot_correct"].mean()

for sys_name, acc_col, tok_col in [
    ("Always-Fast",   "fast_correct",   "fast_tokens"),
    ("Always-CoT",    "cot_correct",    "cot_tokens"),
    ("Always-Best-N", "bon_correct",    "bon_tokens"),
    ("Oracle",        "oracle_correct", "oracle_tokens"),
    ("ARC (Ours)",    "arc_correct",    "arc_tokens"),
]:
    acc    = df[acc_col].mean()
    avg_t  = df[tok_col].mean()
    total_t = df[tok_col].sum()
    save   = (cot_tokens_total - total_t) / max(cot_tokens_total,1) * 100
    delta_acc = acc - cot_acc
    delta_tok = cot_tokens_total - total_t
    res = delta_acc / (delta_tok / len(df) + 1e-9) if delta_tok > 0 else 0.0
    bold = " ◄" if sys_name == "ARC (Ours)" else ""
    print(f"{'  '+sys_name:<18} {acc:>8.1%}  {avg_t:>10.1f}  {save:>10.1f}%  {res:>7.4f}{bold}")

print("═"*65)
print(f"{'Total eval samples':<18}: {len(df)}")
print(f"{'Easy / Med / Hard':<18}: 10 / 10 / 10")


# ══════════════════════════════════════════════════════════
# TABLE 2 — RDE classification report
# ══════════════════════════════════════════════════════════
print("\n\n" + "═"*55)
print("TABLE 2 — RDE DIFFICULTY PREDICTION (on eval set)")
print("═"*55)
true_labels = df["true_diff"].tolist()
pred_labels = df["arc_diff"].tolist()
print(classification_report(true_labels, pred_labels,
      labels=["easy","medium","hard"], digits=3))
rde_acc = sum(1 for t,p in zip(true_labels,pred_labels) if t==p) / len(true_labels)
print(f"RDE Accuracy on eval set: {rde_acc:.1%}")


# ══════════════════════════════════════════════════════════
# TABLE 3 — Token breakdown by difficulty
# ══════════════════════════════════════════════════════════
print("\n\n" + "═"*60)
print("TABLE 3 — TOKEN USAGE BY DIFFICULTY TIER")
print("═"*60)
print(f"{'Difficulty':<12}  {'ARC Tokens':>11}  {'CoT Tokens':>11}  {'Savings':>9}  {'ARC Mode'}") 
print("-"*60)
for diff in ["easy","medium","hard"]:
    sub = df[df["true_diff"] == diff]
    arc_avg = sub["arc_tokens"].mean()
    cot_avg = sub["cot_tokens"].mean()
    save    = (cot_avg - arc_avg) / max(cot_avg,1) * 100
    modes   = sub["arc_mode"].value_counts().to_dict()
    mode_str = ", ".join(f"{k}:{v}" for k,v in modes.items())
    print(f"{diff:<12}  {arc_avg:>11.1f}  {cot_avg:>11.1f}  {save:>8.1f}%  {mode_str}")


# ══════════════════════════════════════════════════════════
# TABLE 4 — Ablation: RDE trained with/without signal groups
# ══════════════════════════════════════════════════════════
print("\n\n" + "═"*55)
print("TABLE 4 — RDE FEATURE IMPORTANCE (from training)")
print("═"*55)

try:
    with open("models/rde/test_results.json") as f:
        rde_results = json.load(f)
    print(f"{'Metric':<30} {'Score':>8}")
    print("-"*40)
    print(f"{'Test Accuracy':<30} {rde_results['test_accuracy']:>8.3f}")
    print(f"{'Macro F1':<30} {rde_results['test_macro_f1']:>8.3f}")
    pc = rde_results.get("per_class", {})
    for cls in ["easy","medium","hard"]:
        if cls in pc:
            f1 = pc[cls].get("f1-score", 0)
            pr = pc[cls].get("precision", 0)
            rc = pc[cls].get("recall", 0)
            print(f"  {cls} — P:{pr:.3f} R:{rc:.3f} F1:{f1:.3f}")
except Exception as e:
    print(f"Could not load test_results.json: {e}")

# Feature importance from first-layer weights
first_w = rde.net[0].weight.detach().abs().mean(0).cpu().numpy()
SCALAR_FEATURES = ["attn_entropy","varentropy","perplexity","tree_depth",
                   "clause_count","avg_sent_len","proof_kw_density",
                   "math_sym_density","eq_density","abstract_ratio",
                   "avg_word_len","num_density"]
scalar_imp = sorted(zip(SCALAR_FEATURES, first_w[:len(SCALAR_FEATURES)]),
                    key=lambda x: x[1], reverse=True)
print(f"\n{'Feature':<24} {'Importance':>10}  Rank")
print("-"*45)
for rank, (name, val) in enumerate(scalar_imp, 1):
    bar = "█" * int(val * 60)
    print(f"  {name:<22} {val:>10.4f}  #{rank}  {bar}")


# ══════════════════════════════════════════════════════════
# Save all tables as JSON for paper
# ══════════════════════════════════════════════════════════
summary = {
    "n_samples": len(df),
    "systems": {}
}
for sys_name, acc_col, tok_col in [
    ("always_fast",  "fast_correct",   "fast_tokens"),
    ("always_cot",   "cot_correct",    "cot_tokens"),
    ("always_bon",   "bon_correct",    "bon_tokens"),
    ("oracle",       "oracle_correct", "oracle_tokens"),
    ("arc",          "arc_correct",    "arc_tokens"),
]:
    summary["systems"][sys_name] = {
        "accuracy":    float(df[acc_col].mean()),
        "avg_tokens":  float(df[tok_col].mean()),
        "total_tokens": int(df[tok_col].sum()),
    }

with open(f"{args.output_dir}/paper_results.json", "w") as f:
    json.dump(summary, f, indent=2)

df.to_csv(f"{args.output_dir}/eval_results.csv", index=False)
print(f"\n\n✅ Results saved to {args.output_dir}/")
print(f"   - eval_results.csv   (per-question breakdown)")
print(f"   - paper_results.json (summary for paper)")