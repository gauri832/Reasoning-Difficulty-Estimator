# scripts/09_upgrade_llm.py
# Upgraded LLM engine for ARC pipeline.
# Fixes: 'GenerationConfig' clone error with Qwen2.5 + old transformers.
#
# Run:
#   python scripts/09_upgrade_llm.py --backend hf_small
#   python scripts/09_upgrade_llm.py --backend ollama

import argparse
import copy
import os
import re
import sys
import time

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


if not hasattr(GenerationConfig, "clone"):
    def _compat_clone(self):
        return copy.deepcopy(self)

    GenerationConfig.clone = _compat_clone

# ── CLI args ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--backend",      default="hf_small",
                    choices=["hf_small","ollama","hf_api"])
parser.add_argument("--hf-model",     default="Qwen/Qwen2.5-0.5B-Instruct")
parser.add_argument("--ollama-model", default="llama3.2")
parser.add_argument("--no-interactive", action="store_true")
args = parser.parse_args()

BACKEND      = args.backend
HF_MODEL     = args.hf_model
OLLAMA_MODEL = args.ollama_model
OLLAMA_URL   = "http://localhost:11434/api/generate"
HF_API_TOKEN = os.environ.get("HF_TOKEN", "")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN      = 256

print(f"\nBackend : {BACKEND}")
print(f"Device  : {DEVICE}")

import transformers
print(f"transformers: {transformers.__version__}")


# ══════════════════════════════════════════════════════════
# Prompt builder
# ══════════════════════════════════════════════════════════
COT_PREFIX = (
    "You are a concise, precise reasoning assistant.\n"
    "Think step by step, then give your final answer.\n\n"
)

def build_prompt(question: str, mode: str, tok=None) -> str:
    """
    For instruction-tuned models (Qwen, Llama-Instruct etc.) use the
    chat template when available — it gives much better answers.
    Falls back to plain text for base models.
    """
    if tok is not None and hasattr(tok, "apply_chat_template"):
        try:
            if mode == "fast":
                system = "Answer in one sentence or less. Be direct and concise."
                user   = question
            else:
                system = (
                    "You are a precise reasoning assistant. "
                    "Think step by step, show your work clearly, "
                    "then state the final answer explicitly."
                )
                user = question

            messages = [
                {"role": "system",    "content": system},
                {"role": "user",      "content": user},
            ]
            return tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass   # fall through to plain text

    # Plain-text fallback (base models, distilgpt2, etc.)
    if mode == "fast":
        return f"Answer concisely.\nQ: {question}\nA:"
    return f"{COT_PREFIX}Question: {question}\n\nStep-by-step:"


# ══════════════════════════════════════════════════════════
# Generation backends
# ══════════════════════════════════════════════════════════
def _mode_params(mode: str) -> dict:
    params = {
        "fast":      {"temperature": 0.1, "max_new_tokens": 150, "do_sample": False},
        "cot":       {"temperature": 0.5, "max_new_tokens": 512, "do_sample": True},
        "best_of_n": {"temperature": 0.8, "max_new_tokens": 600, "do_sample": True},
    }.get(mode, {"temperature": 0.5, "max_new_tokens": 300, "do_sample": True})
    if DEVICE == "cpu":
        params = dict(params)
        params["do_sample"] = False
        params["temperature"] = 0.0
        params["max_new_tokens"] = min(params["max_new_tokens"], 64)
    return params


def generate_hf_small(prompt: str, mode: str, tok, mdl) -> tuple:
    """
    KEY FIX for 'GenerationConfig has no attribute clone':
    Reset the model's generation_config to a plain default before
    calling .generate(). Qwen2.5 ships a custom GenerationConfig
    that breaks with transformers < 4.43.
    """
    p = _mode_params(mode)

    # Build clean generation config — avoids the clone() issue entirely
    clean_cfg = GenerationConfig(
        pad_token_id  = tok.eos_token_id,
        eos_token_id  = tok.eos_token_id,
        bos_token_id  = getattr(tok, "bos_token_id", tok.eos_token_id),
        max_new_tokens= p["max_new_tokens"],
        do_sample     = p["do_sample"],
    )
    if p["do_sample"]:
        clean_cfg.temperature = p["temperature"]
        clean_cfg.top_p       = 0.92
    clean_cfg.no_repeat_ngram_size = 3

    try:
        mdl.generation_config = clean_cfg
    except Exception:
        pass

    inputs = tok(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    ).to(DEVICE)

    try:
        with torch.no_grad():
            out = mdl.generate(
                input_ids      = inputs["input_ids"],
                attention_mask = inputs["attention_mask"],
                generation_config = clean_cfg,
            )
        new_ids = out[0][inputs["input_ids"].shape[1]:]
        answer  = tok.decode(new_ids, skip_special_tokens=True).strip()
        return answer, len(new_ids)
    except Exception as e:
        return f"[HF generation error: {e}]", 0


def generate_ollama(prompt: str, mode: str) -> tuple:
    import requests
    p = _mode_params(mode)
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": p["temperature"],
            "num_predict": p["max_new_tokens"],
        }
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        data = r.json()
        answer = data.get("response","").strip()
        return answer, data.get("eval_count", len(answer.split()))
    except Exception as e:
        return f"[Ollama error: {e}]", 0


def generate_hf_api(prompt: str, mode: str) -> tuple:
    import requests
    p = _mode_params(mode)
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": p["temperature"],
            "max_new_tokens": 300,
            "return_full_text": False,
        }
    }
    url = f"https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        data = r.json()
        answer = data[0].get("generated_text","") if isinstance(data,list) else str(data)
        return answer.strip(), len(answer.split())
    except Exception as e:
        return f"[HF API error: {e}]", 0


def generate_answer(prompt: str, mode: str) -> tuple:
    if BACKEND == "hf_small":
        return generate_hf_small(prompt, mode, hf_tok, hf_model)
    elif BACKEND == "ollama":
        return generate_ollama(prompt, mode)
    elif BACKEND == "hf_api":
        return generate_hf_api(prompt, mode)
    return "[Unknown backend]", 0


# ══════════════════════════════════════════════════════════
# RDE + signal extractor (distilgpt2 for features only)
# ══════════════════════════════════════════════════════════
nlp = spacy.load("en_core_web_sm")

sig_tok = AutoTokenizer.from_pretrained("distilgpt2")
if sig_tok.pad_token is None:
    sig_tok.pad_token = sig_tok.eos_token
sig_lm = AutoModelForCausalLM.from_pretrained(
    "distilgpt2").to(DEVICE)
sig_lm.eval()

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
print("RDE loaded.")

PROOF_KW = {"prove","proof","show that","theorem","lemma","contradiction",
            "induction","iff","derive","irrational","integer solutions"}
MATH_RE  = re.compile(r'[\^∑√∞≤≥≠]|\\[a-zA-Z]+|\d+\^\d+|[a-z]\([a-z]\)')
ABS_N    = {"theorem","lemma","polynomial","integer","irrational",
            "prime","matrix","eigenvalue","derivative"}

def _math_features(text):
    tl   = text.lower()
    toks = tl.split()
    n    = max(len(toks), 1)
    pk = sum(1 for kw in PROOF_KW if kw in tl) / n
    ms = len(MATH_RE.findall(text)) / n
    eq = len(re.findall(r'[a-zA-Z]\^?\d+|[a-zA-Z]\([a-zA-Z0-9]+\)', text)) / n
    ar = sum(1 for t in toks if t.strip('.,;:') in ABS_N) / n
    cl = [t.strip('.,;:!?()') for t in toks if t.strip('.,;:!?()')]
    aw = float(np.mean([len(t) for t in cl])) if cl else 0.0
    nd = sum(1 for t in toks if re.match(r'^\d+\.?\d*$', t)) / n
    return pk, ms, eq, ar, aw, nd

def extract_features(text):
    inputs = sig_tok(text, return_tensors="pt", max_length=MAX_LEN,
                     truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        out = sig_lm(**inputs, labels=inputs["input_ids"],
                     output_hidden_states=True, output_attentions=True)
    emb      = out.hidden_states[-1].mean(1).squeeze().cpu().numpy()
    ents     = [(-torch.sum((la.squeeze(0)+1e-9)*torch.log(la.squeeze(0)+1e-9),
                            dim=-1)).mean().item()
                for la in (out.attentions or [])]
    attn_ent = float(np.mean(ents)) if ents else 0.0
    logits   = out.logits.squeeze(0)
    probs    = torch.softmax(logits,-1) + 1e-9
    var_ent  = float((-torch.sum(probs*torch.log(probs),-1)).var().item())
    ppl      = float(torch.exp(out.loss).item())
    doc      = nlp(text[:1000])
    sents    = list(doc.sents)
    max_d    = 0
    for tok2 in doc:
        d, t = 0, tok2
        while t.head != t: t = t.head; d += 1
        max_d = max(max_d, d)
    cl_cnt = sum(1 for t in doc if t.dep_ in ("relcl","advcl","ccomp","xcomp"))
    avg_sl = float(np.mean([len(s) for s in sents])) if sents else 0.0
    pk, ms, eq, ar, aw, nd = _math_features(text)
    scalars = np.array([attn_ent, var_ent, ppl, float(max_d),
                        float(cl_cnt), avg_sl, pk, ms, eq, ar, aw, nd],
                       dtype=np.float32)
    return scalars, emb

HARD_CUES   = re.compile(
    r"\b(prove|proof|contradiction|induction|theorem|lemma|corollary|"
    r"irrational|integer solutions?|recurrence|show that|if and only if|iff)\b",
    re.IGNORECASE,
)
MEDIUM_CUES = re.compile(
    r"\b(explain|difference between|compare|complexity|time complexity|algorithm|"
    r"average speed|merge sort|quicksort|queue|stack)\b",
    re.IGNORECASE,
)
EASY_ARITH  = re.compile(
    r"^\s*(what is|calculate|compute|find)\b.{0,80}(\d+\s*[+\-*/]\s*\d+)",
    re.IGNORECASE,
)
EASY_FACT   = re.compile(
    r"\b(capital of|what is the capital|who is|when did|define)\b",
    re.IGNORECASE,
)

def _rule_override(question: str):
    """Exact same rule-override used in 06_test_rde.py for consistency."""
    q = question.lower()
    if HARD_CUES.search(q):   return "hard"
    if MEDIUM_CUES.search(q): return "medium"
    if EASY_ARITH.search(q) or EASY_FACT.search(q): return "easy"
    return None

def rde_predict(question):
    scalars, emb = extract_features(question)
    sn = (scalars - sc_mean)  / (sc_scale + 1e-8)
    en = (np.array(emb, dtype=np.float32) - em_mean) / (em_scale + 1e-8)
    x  = torch.tensor(np.concatenate([sn, en]),
                      dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(rde(x), -1).squeeze().cpu().numpy()

    # Apply same rule-override calibration as 06_test_rde.py
    override = _rule_override(question)
    if override and override in classes:
        idx_ov = classes.index(override)
        if probs[idx_ov] < 0.60:
            probs[idx_ov] = 0.60
            probs = probs / probs.sum()
        idx = idx_ov
    else:
        idx = int(probs.argmax())

    return classes[idx], float(probs[idx]), dict(zip(classes, probs.tolist()))

def select_mode(difficulty, rde_conf):
    mode = {"easy":"fast","medium":"cot","hard":"best_of_n"}.get(difficulty,"cot")
    escalated = False
    if rde_conf < 0.55:
        if mode == "fast":   mode = "cot";       escalated = True
        elif mode == "cot":  mode = "best_of_n"; escalated = True
    return mode, escalated


# ── Load answer LLM ───────────────────────────────────────
hf_tok = hf_model = None

if BACKEND == "hf_small":
    print(f"Loading {HF_MODEL} ...")
    hf_tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    hf_model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        dtype=torch.float32,
        trust_remote_code=True,
    ).to(DEVICE)
    hf_model.eval()
    print("HF small model ready.")

elif BACKEND == "ollama":
    import requests as _r
    try:
        _r.get("http://localhost:11434", timeout=3)
        print(f"[OK] Ollama running | model: {OLLAMA_MODEL}")
    except:
        print("[ERROR] Ollama not running. Start: ollama serve")
        print(f"   Pull model: ollama pull {OLLAMA_MODEL}")
        sys.exit(1)

elif BACKEND == "hf_api":
    if not HF_API_TOKEN:
        print("[ERROR] Set HF_TOKEN env variable first.")
        sys.exit(1)


# ── Full pipeline ─────────────────────────────────────────
def run(question: str) -> dict:
    t0 = time.time()
    difficulty, rde_conf, _ = rde_predict(question)
    mode, escalated         = select_mode(difficulty, rde_conf)
    # Pass hf_tok so chat template is used for instruction-tuned models
    answer_tok = hf_tok if BACKEND == "hf_small" else None
    prompt     = build_prompt(question, mode, tok=answer_tok)
    answer, tokens = generate_answer(prompt, mode)
    return {
        "question":   question,
        "difficulty": difficulty,
        "rde_conf":   rde_conf,
        "mode":       mode,
        "escalated":  escalated,
        "answer":     answer,
        "tokens":     tokens,
        "time":       round(time.time() - t0, 2),
    }


# ── Benchmark ─────────────────────────────────────────────
benchmark = [
    ("What is 15 + 27?",                                              "easy"),
    ("Write a Python function to reverse a string.",                  "easy"),
    ("What is the capital of France?",                                "easy"),
    ("Explain the difference between TCP and UDP.",                   "medium"),
    ("What is the average speed if 60 km/h for 2h then 80 for 3h?",  "medium"),
    ("What is the time complexity of merge sort? Explain.",           "medium"),
    ("Prove that there are infinitely many prime numbers.",            "hard"),
    ("Prove that sqrt(2) is irrational.",                             "hard"),
    ("Solve the recurrence T(n) = 2T(n/2) + n log n.",               "hard"),
]

if BACKEND == "hf_small" and DEVICE == "cpu":
    benchmark = benchmark[:6]
    print("CPU mode: running 6-question benchmark.")

print("\n" + "=" * 76)
print(f"{'Question':<44} {'Diff':>6}  {'Mode':>10}  {'Tok':>4}  {'OK':>2}")
print("=" * 76)

total_tokens = 0
correct_diff = 0

for q, exp in benchmark:
    try:
        r = run(q)
        ok = "OK" if r["difficulty"] == exp else "XX"
        if r["difficulty"] == exp: correct_diff += 1
        total_tokens += r["tokens"]
        print(f"{ok} {q[:42]:<42} {r['difficulty']:>6}  {r['mode']:>10}  {r['tokens']:>4}")
        # Print first 120 chars of answer, indented
        ans_preview = r["answer"].replace("\n"," ")[:120]
        print(f"   -> {ans_preview}")
    except Exception as e:
        print(f"XX {q[:42]:<42}  ERROR: {e}")
    print()

print("=" * 76)
baseline = 512 * len(benchmark)
savings  = (baseline - total_tokens) / baseline * 100
print(f"\nDifficulty accuracy : {correct_diff}/{len(benchmark)}")
print(f"Total tokens        : {total_tokens}  (always-CoT est: {baseline})")
print(f"Token savings (RES) : {savings:.1f}%")

# ── Interactive ───────────────────────────────────────────
if not args.no_interactive:
    print("\nInteractive mode (type 'quit' to exit)")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("quit","exit","q"): break
        if not q: continue
        try:
            r = run(q)
            tag = f"[{r['difficulty'].upper()} | {r['mode']} | {r['rde_conf']:.0%}]"
            tag += " [escalated]" if r["escalated"] else ""
            print(f"  {tag}")
            print(f"  Tokens: {r['tokens']} | Time: {r['time']}s")
            print(f"  Answer:\n{r['answer'][:500]}")
        except Exception as e:
            print(f"  Error: {e}")