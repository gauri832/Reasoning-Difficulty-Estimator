# scripts/08_full_pipeline.py
# End-to-end: Question -> RDE -> ARC -> Answer

import importlib.util
import os
import re
import sys
import time

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN = 256

nlp = spacy.load("en_core_web_sm")


def load_arc_controller_class():
    """Load ARCController from scripts/07_arc_controller.py."""
    try:
        from arc_controller import ARCController  # type: ignore
        return ARCController
    except ModuleNotFoundError:
        pass

    this_dir = os.path.dirname(__file__)
    arc_path = os.path.join(this_dir, "07_arc_controller.py")
    spec = importlib.util.spec_from_file_location("arc_controller", arc_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load ARC controller from {arc_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["arc_controller"] = mod
    spec.loader.exec_module(mod)
    return mod.ARCController


ARCController = load_arc_controller_class()


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
rde_model = RDEClassifier(
    ckpt["input_dim"], ckpt["hidden_dims"], ckpt["num_classes"], ckpt["dropout"]
).to(DEVICE)
rde_model.load_state_dict(ckpt["model_state"])
rde_model.eval()

sc_mean = np.load("models/rde/scalar_mean.npy")
sc_scale = np.load("models/rde/scalar_scale.npy")
em_mean = np.load("models/rde/emb_mean.npy")
em_scale = np.load("models/rde/emb_scale.npy")
classes = list(ckpt["label_classes"])

# Signal extractor model (for runtime feature extraction)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
lm = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
lm.eval()

print("RDE loaded")

PROOF_KEYWORDS = {
    "prove", "proof", "show that", "demonstrate", "derive",
    "theorem", "lemma", "corollary", "iff", "if and only if",
    "contradiction", "induction", "therefore", "hence", "assume"
}
MATH_SYM_RE = re.compile(
    r"[\^\u2200\u2203\u2208\u2211\u221a\u221e\u2264\u2265\u2260\u2261\u00b1\u00d7\u00f7]|\\[a-zA-Z]+|\d+\^\d+|[a-z]\([a-z]\)"
)
ABSTRACT_NOUNS = {
    "theorem", "lemma", "corollary", "proof", "polynomial", "integer",
    "rational", "irrational", "prime", "matrix", "eigenvalue", "derivative"
}


def math_features(text: str):
    tl = text.lower()
    tokens = tl.split()
    n = max(len(tokens), 1)

    proof_kw = sum(1 for kw in PROOF_KEYWORDS if kw in tl) / n
    math_sym = len(MATH_SYM_RE.findall(text)) / n
    eq_dens = len(re.findall(r"[a-zA-Z]\^?\d+|[a-zA-Z]_\{?[a-zA-Z0-9]+\}?", text)) / n
    abs_ratio = sum(1 for t in tokens if t.strip(".,;:") in ABSTRACT_NOUNS) / n

    clean = [t.strip(".,;:!?()") for t in tokens if t.strip(".,;:!?()")]
    avg_wlen = float(np.mean([len(t) for t in clean])) if clean else 0.0
    num_dens = sum(1 for t in tokens if re.match(r"^\d+\.?\d*$", t)) / n
    return proof_kw, math_sym, eq_dens, abs_ratio, avg_wlen, num_dens


def extract_features(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        truncation=True,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        out = lm(**inputs, output_hidden_states=True, output_attentions=True)

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
    var_ent = float((-torch.sum(probs * torch.log(probs), dim=-1)).var().item())

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

    cl_cnt = sum(1 for t in doc if t.dep_ in ("relcl", "advcl", "ccomp", "xcomp"))
    avg_sl = float(np.mean([len(s) for s in sents])) if sents else 0.0

    pk, ms, eq, ar, aw, nd = math_features(text)
    scalars = np.array(
        [attn_ent, var_ent, ppl, float(max_d), float(cl_cnt), avg_sl, pk, ms, eq, ar, aw, nd],
        dtype=np.float32,
    )
    return scalars, emb


def rde_predict(question: str):
    scalars, emb = extract_features(question)
    s_norm = (scalars - sc_mean) / (sc_scale + 1e-8)
    e_norm = (np.array(emb, dtype=np.float32) - em_mean) / (em_scale + 1e-8)
    x = torch.tensor(np.concatenate([s_norm, e_norm]), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        probs = torch.softmax(rde_model(x), dim=-1).squeeze().cpu().numpy()

    idx = int(np.argmax(probs))
    return classes[idx], float(probs[idx]), dict(zip(classes, probs.tolist()))


arc = ARCController(model_name=MODEL_NAME, device=DEVICE)


def run_pipeline(question: str) -> dict:
    t0 = time.time()

    difficulty, rde_conf, rde_probs = rde_predict(question)
    decision = arc.select_mode(difficulty, rde_conf)
    result = arc.generate(question, difficulty, rde_conf)

    total_time = time.time() - t0

    return {
        "question": question,
        "difficulty": difficulty,
        "rde_confidence": rde_conf,
        "rde_probs": rde_probs,
        "mode_used": result.mode_used,
        "escalated": decision.escalated,
        "answer": result.answer,
        "tokens_generated": result.tokens_generated,
        "backtracks": result.backtracks,
        "time_total": round(total_time, 2),
        "time_generation": result.time_seconds,
    }


benchmark = [
    ("What is 8 x 7?", "easy"),
    ("Write a function to find the max in a list.", "easy"),
    ("What is the capital of Australia?", "easy"),
    ("Explain the difference between TCP and UDP.", "medium"),
    ("What is the average speed if 60 km/h for 2h, 80 for 3h?", "medium"),
    ("Describe how merge sort works and its complexity.", "medium"),
    ("Prove that there are infinitely many prime numbers.", "hard"),
    ("Prove that sqrt(2) is irrational.", "hard"),
    ("Find all solutions to x^2 == 1 (mod 8).", "hard"),
]

print("\n" + "=" * 78)
print(f"{'Question':<38} {'Diff':>6}  {'Mode':>10}  {'Tok':>4}  {'BT':>3}  {'OK':>2}")
print("=" * 78)

correct_diff = 0
total_tokens = 0
total_time = 0.0

for q, expected_diff in benchmark:
    trace = run_pipeline(q)
    ok = "OK" if trace["difficulty"] == expected_diff else "X"
    if trace["difficulty"] == expected_diff:
        correct_diff += 1

    total_tokens += trace["tokens_generated"]
    total_time += trace["time_total"]

    print(
        f"{ok:>2} {q[:36]:<36} {trace['difficulty']:>6}  "
        f"{trace['mode_used']:>10}  {trace['tokens_generated']:>4}  {trace['backtracks']:>3}"
    )

print("=" * 78)
print(f"\nDifficulty accuracy : {correct_diff}/{len(benchmark)}")
print(f"Total tokens used   : {total_tokens}")
print(f"Avg tokens/query    : {total_tokens / len(benchmark):.1f}")
print(f"Total time          : {total_time:.1f}s")
print(f"Avg time/query      : {total_time / len(benchmark):.1f}s")

always_cot_tokens = 80 * len(benchmark)
our_tokens = total_tokens
token_savings_pct = (always_cot_tokens - our_tokens) / max(always_cot_tokens, 1) * 100

print("\nReasoning Efficiency Score (RES)")
print(f"Always-CoT tokens (est.): {always_cot_tokens}")
print(f"ARC tokens used         : {our_tokens}")
print(f"Token savings           : {token_savings_pct:.1f}%")

print("\n\nInteractive Full Pipeline (type 'quit' to exit)")
while True:
    q = input("\nAsk anything: ").strip()
    if q.lower() in ("quit", "exit", "q"):
        break
    if not q:
        continue

    trace = run_pipeline(q)
    print(f"\n  Difficulty  : {trace['difficulty'].upper()} (RDE conf: {trace['rde_confidence']:.1%})")
    print(f"  Mode used   : {trace['mode_used']}" + (" [escalated]" if trace['escalated'] else ""))
    print(f"  Backtracks  : {trace['backtracks']}")
    print(f"  Tokens used : {trace['tokens_generated']}")
    print(f"  Time        : {trace['time_total']}s")
    print(f"\n  Answer: {trace['answer'][:300]}")
