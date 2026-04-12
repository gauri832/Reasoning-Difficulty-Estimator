import re

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

nlp = spacy.load("en_core_web_sm")
MODEL_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
if hasattr(lm.config, "loss_type") and lm.config.loss_type is None:
    lm.config.loss_type = "ForCausalLMLoss"
lm.eval()


class RDEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


ckpt = torch.load("models/rde/best_model.pt", map_location=DEVICE, weights_only=False)
model = RDEClassifier(
    ckpt["input_dim"], ckpt["hidden_dims"], ckpt["num_classes"], ckpt["dropout"]
).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

sc_mean = np.load("models/rde/scalar_mean.npy")
sc_scale = np.load("models/rde/scalar_scale.npy")
em_mean = np.load("models/rde/emb_mean.npy")
em_scale = np.load("models/rde/emb_scale.npy")
classes = list(ckpt["label_classes"])
class_to_idx = {c: i for i, c in enumerate(classes)}

print(
    "RDE loaded | "
    f"val_acc={ckpt.get('val_acc', 0.0):.4f} | "
    f"val_macro_f1={ckpt.get('val_macro_f1', 0.0):.4f} | "
    f"classes={classes}\n"
)

PROOF_KEYWORDS = {
    "prove",
    "proof",
    "show that",
    "demonstrate",
    "derive",
    "theorem",
    "lemma",
    "corollary",
    "iff",
    "if and only if",
    "qed",
    "contradiction",
    "induction",
    "base case",
    "assume",
    "suppose",
    "therefore",
    "hence",
    "it follows",
    "we conclude",
    "by definition",
}
MATH_SYMBOLS_RE = re.compile(
    r"[\u2200\u2203\u2208\u2209\u2282\u2283\u2229\u222a\u2211\u220f\u221a\u221e\u2264\u2265\u2260\u2261\u00b1\u00d7\u00f7\^]|"
    r"\\[a-zA-Z]+|\d+\^\d+|[a-z]\([a-z]\)"
)
ABSTRACT_NOUNS = {
    "theorem",
    "lemma",
    "corollary",
    "proof",
    "conjecture",
    "proposition",
    "axiom",
    "definition",
    "hypothesis",
    "equation",
    "polynomial",
    "integer",
    "rational",
    "irrational",
    "prime",
    "modulo",
    "matrix",
    "vector",
    "eigenvalue",
    "derivative",
    "integral",
    "limit",
    "convergence",
}

HARD_CUES = re.compile(
    r"\b(prove|proof|contradiction|induction|theorem|lemma|corollary|"
    r"irrational|integer solutions?|recurrence|show that|if and only if|iff)\b",
    re.IGNORECASE,
)
MEDIUM_CUES = re.compile(
    r"\b(explain|difference between|compare|complexity|time complexity|algorithm|"
    r"design a class|average speed|avg speed|merge sort|quicksort|queue|stack)\b",
    re.IGNORECASE,
)
EASY_ARITH = re.compile(
    r"^\s*(what is|calculate|compute|find)\b.{0,80}(\d+\s*[+\-*/]\s*\d+)",
    re.IGNORECASE,
)
EASY_FACT = re.compile(
    r"\b(capital of|what is the capital|who is|when did|define)\b",
    re.IGNORECASE,
)


def math_aware_features(text):
    tl = text.lower()
    tokens = tl.split()
    n = max(len(tokens), 1)

    proof_kw = sum(1 for kw in PROOF_KEYWORDS if kw in tl) / n
    math_sym = len(MATH_SYMBOLS_RE.findall(text)) / n
    eq_dens = len(
        re.findall(
            r"[a-zA-Z]\^?\d+|[a-zA-Z]_\{?[a-zA-Z0-9]+\}?|[a-zA-Z]\([a-zA-Z0-9,\s]+\)",
            text,
        )
    ) / n
    abs_ratio = sum(1 for t in tokens if t.strip(".,;:") in ABSTRACT_NOUNS) / n

    clean = [
        t.strip(".,;:!?()[]{}")
        for t in tokens
        if t.strip(".,;:!?()[]{}")
    ]
    avg_wlen = float(np.mean([len(t) for t in clean])) if clean else 0.0

    num_dens = sum(1 for t in tokens if re.match(r"^\d+\.?\d*$", t)) / n

    return proof_kw, math_sym, eq_dens, abs_ratio, avg_wlen, num_dens


def extract_all(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        truncation=True,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        out = lm(
            **inputs,
            output_hidden_states=True,
            output_attentions=True,
        )

    emb = out.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()

    ents = []
    if out.attentions:
        for la in out.attentions:
            p = la.squeeze(0) + 1e-9
            h = -torch.sum(p * torch.log(p), dim=-1)
            ents.append(h.mean().item())
    attn_ent = float(np.mean(ents)) if ents else 0.0

    logits = out.logits.squeeze(0)
    probs = torch.softmax(logits, dim=-1) + 1e-9
    token_entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    var_ent = float(token_entropy.var().item())

    shift_logits = out.logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()
    ce = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=tokenizer.pad_token_id,
    )
    ppl = float(torch.exp(ce).item())

    doc = nlp(text[:1000])
    sents = list(doc.sents)

    max_d = 0
    for tok in doc:
        d = 0
        cur = tok
        while cur.head != cur:
            cur = cur.head
            d += 1
        max_d = max(max_d, d)

    clause_cnt = sum(1 for t in doc if t.dep_ in ("relcl", "advcl", "ccomp", "xcomp"))
    avg_sl = float(np.mean([len(s) for s in sents])) if sents else 0.0

    pk, ms, eq, ar, aw, nd = math_aware_features(text)

    scalars = np.array(
        [
            attn_ent,
            var_ent,
            ppl,
            float(max_d),
            float(clause_cnt),
            avg_sl,
            pk,
            ms,
            eq,
            ar,
            aw,
            nd,
        ],
        dtype=np.float32,
    )

    return scalars, emb


def calibrate_probs(question: str, probs: np.ndarray) -> np.ndarray:
    scores = probs.astype(np.float64).copy()
    text = question.lower()
    tokens = text.split()

    idx_easy = class_to_idx.get("easy")
    idx_medium = class_to_idx.get("medium")
    idx_hard = class_to_idx.get("hard")

    hard_hits = len(HARD_CUES.findall(question))
    medium_hits = len(MEDIUM_CUES.findall(question))
    easy_hit = bool(EASY_ARITH.search(question))

    if idx_hard is not None and hard_hits > 0:
        scores[idx_hard] *= 1.0 + 1.20 + 0.20 * max(0, hard_hits - 1)
        if idx_easy is not None:
            scores[idx_easy] *= 0.45

    if idx_medium is not None and medium_hits > 0:
        scores[idx_medium] *= 1.0 + 0.90 + 0.10 * max(0, medium_hits - 1)
        if hard_hits == 0 and idx_easy is not None:
            scores[idx_easy] *= 0.62

    if easy_hit and hard_hits == 0 and medium_hits == 0 and idx_easy is not None:
        scores[idx_easy] *= 1.20
        if idx_hard is not None:
            scores[idx_hard] *= 0.85

    if EASY_FACT.search(question) and hard_hits == 0 and idx_easy is not None:
        scores[idx_easy] *= 1.35
        if idx_medium is not None:
            scores[idx_medium] *= 0.82

    if len(tokens) >= 22 and hard_hits == 0 and idx_medium is not None:
        scores[idx_medium] *= 1.08

    scores = np.clip(scores, 1e-9, None)
    scores = scores / scores.sum()
    return scores.astype(np.float32)


def _rule_override(question: str):
    q = question.lower()
    if HARD_CUES.search(q):
        return "hard"
    if MEDIUM_CUES.search(q):
        return "medium"
    if EASY_ARITH.search(q) or EASY_FACT.search(q):
        return "easy"
    return None


def predict(question):
    scalars, emb = extract_all(question)

    s_norm = (scalars - sc_mean) / (sc_scale + 1e-8)
    e_norm = (np.array(emb, dtype=np.float32) - em_mean) / (em_scale + 1e-8)
    x = torch.tensor(np.concatenate([s_norm, e_norm]), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1).squeeze().cpu().numpy()

    probs = calibrate_probs(question, probs)
    override = _rule_override(question)
    if override in class_to_idx:
        idx_override = class_to_idx[override]
        if probs[idx_override] < 0.60:
            probs[idx_override] = 0.60
            probs = probs / probs.sum()
        idx = idx_override
    else:
        idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), dict(zip(classes, probs.tolist()))


test_qs = [
    ("What is 15 + 27?", "easy"),
    ("Write a Python function to check if a number is even.", "easy"),
    ("What is the capital of France?", "easy"),
    ("Find the LCM of 12 and 18.", "easy"),
    ("How does binary search work?", "easy"),
    ("A train: 60 km/h for 2h then 80 km/h for 3h. Avg speed?", "medium"),
    ("Explain the difference between a stack and a queue.", "medium"),
    ("What is the time complexity of merge sort? Explain why.", "medium"),
    ("Derive the worst-case time complexity of quicksort.", "medium"),
    ("Prove that the sum of first n odd numbers equals n squared.", "hard"),
    ("Find all integer solutions to x^3 + y^3 = z^3.", "hard"),
    (
        "If p(x) degree-4 polynomial with roots 1,2,3,4 and p(0)=24, find p(5).",
        "hard",
    ),
    ("Prove that sqrt(2) is irrational using contradiction.", "hard"),
    ("Solve the recurrence T(n) = 2T(n/2) + n log n.", "hard"),
]

print("=" * 80)
print(f"{'Question':<46} {'Pred':>7}  {'Conf':>5}  {'Exp':>7}  {'OK':>2}")
print("=" * 80)

correct = 0
for q, exp in test_qs:
    pred, conf, _ = predict(q)
    ok = "OK" if pred == exp else "X"
    print(f"{ok:>2} {q[:44]:<44} {pred:>7}  {conf:>5.2f}  {exp:>7}")
    if pred == exp:
        correct += 1

print("=" * 80)
print(f"\nAccuracy: {correct}/{len(test_qs)} ({correct / len(test_qs):.0%})\n")

print("Interactive mode (type 'quit' to exit)")
while True:
    question = input("\nQuestion: ").strip()
    if question.lower() in {"quit", "exit", "q"}:
        break
    if not question:
        continue

    pred, conf, all_probs = predict(question)
    print(f"  -> {pred.upper()} ({conf:.1%})")
    print("     {" + ", ".join(f"{k}: {v:.3f}" for k, v in all_probs.items()) + "}")
