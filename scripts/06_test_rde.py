# scripts/06_test_rde.py
# Quick sanity test of the trained RDE on new questions.
# Run after 05_train_rde.py completes.

import torch
import numpy as np
import spacy
from transformers import AutoTokenizer, AutoModelForCausalLM

nlp = spacy.load("en_core_web_sm")

MODEL_NAME = "distilgpt2"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LEN    = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

lm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, output_attentions=True).to(DEVICE)
lm.eval()


# ── Rebuild RDE model ─────────────────────────────────────
import torch.nn as nn

class RDEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev,h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


ckpt = torch.load(
    "models/rde/best_model.pt",
    map_location=DEVICE,
    weights_only=False   # 🔥 REQUIRED FIX
)
input_dim = ckpt["input_dim"]   # 🔥 use saved value

model = RDEClassifier(
    input_dim=input_dim,
    hidden_dims=ckpt["hidden_dims"],
    num_classes=ckpt["num_classes"],
    dropout=ckpt["dropout"]
).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

scaler_mean  = np.load("models/rde/scaler_mean.npy")
emb_mean  = np.load("models/rde/emb_mean.npy")
emb_scale = np.load("models/rde/emb_scale.npy")
scaler_scale = np.load("models/rde/scaler_scale.npy")
classes      = ckpt["label_classes"]   # ['easy', 'hard', 'medium']

def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=MAX_LEN,
        truncation=True,
        padding=True
    ).to(DEVICE)

    with torch.no_grad():
        out = lm(**inputs, output_hidden_states=True)

    hidden = out.hidden_states[-1]
    return hidden.mean(dim=1).squeeze().cpu().numpy()


# ── Feature extraction (same as Phase 1) ──────────────────
def extract_features(text):
    inputs = tokenizer(text, return_tensors="pt",
                       max_length=MAX_LEN, truncation=True,
                       padding=True).to(DEVICE)
    with torch.no_grad():
        out = lm(**inputs, labels=inputs["input_ids"],
                 output_hidden_states=True)

    # attn entropy
    entropies = []
    if out.attentions:
        for la in out.attentions:
            p = la.squeeze(0) + 1e-9
            H = -torch.sum(p * torch.log(p), dim=-1)
            entropies.append(H.mean().item())
    attn_ent = float(np.mean(entropies)) if entropies else 0.0

    # varentropy
    logits   = out.logits.squeeze(0)
    probs    = torch.softmax(logits, dim=-1) + 1e-9
    entropy  = -torch.sum(probs * torch.log(probs), dim=-1)
    var_ent  = float(entropy.var().item())

    # perplexity
    ppl = float(torch.exp(out.loss).item())

    # syntax
    doc  = nlp(text[:1000])
    sents = list(doc.sents)
    max_depth = 0
    for token in doc:
        d, t = 0, token
        while t.head != t: t = t.head; d += 1
        max_depth = max(max_depth, d)
    clause_count = sum(1 for t in doc
                       if t.dep_ in ("relcl","advcl","ccomp","xcomp"))
    avg_slen = float(np.mean([len(s) for s in sents])) if sents else 0.0

    return np.array([attn_ent, var_ent, ppl,
                     float(max_depth), float(clause_count), avg_slen],
                    dtype=np.float32)


def predict_difficulty(question):
   # scalar features
    scalar_feats = extract_features(question)
    scalar_scaled = (scalar_feats - scaler_mean) / scaler_scale

    # embedding
    emb_raw = get_embedding(question)
    emb_scaled = (np.array(emb_raw, dtype=np.float32) - emb_mean) / (emb_scale + 1e-8)

    # combine
    features = np.concatenate([scalar_scaled, emb_scaled])
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=-1).squeeze().cpu().numpy()
    pred_idx    = probs.argmax()
    pred_label  = classes[pred_idx]
    confidence  = float(probs[pred_idx])
    return pred_label, confidence, dict(zip(classes, probs.tolist()))


# ── Test on sample questions ───────────────────────────────
test_questions = [
    # Expected: easy
    ("What is 15 + 27?",                                    "easy"),
    ("Write a function to check if a number is even.",      "easy"),
    ("What is the capital of France?",                      "easy"),

    # Expected: medium
    ("A train travels 60 mph for 2 hours then 80 mph for 3 hours. "
     "What is the average speed for the whole journey?",    "medium"),
    ("Explain the difference between a stack and a queue "
     "and give a use case for each.",                       "medium"),

    # Expected: hard
    ("Prove that the sum of the first n odd numbers equals n squared.",  "hard"),
    ("Find all integer solutions to x^3 + y^3 = z^3.",                  "hard"),
    ("If p(x) is a polynomial of degree 4 with roots at "
     "x=1,2,3,4 and p(0)=24, find p(5).",                               "hard"),
]

print("=" * 65)
print(f"{'Question':<45} {'Pred':>6}  {'Conf':>5}  {'Expected':>8}")
print("=" * 65)

correct = 0
for q, expected in test_questions:
    pred, conf, all_probs = predict_difficulty(q)
    ok = "✅" if pred == expected else "❌"
    print(f"{ok} {q[:42]:<42} {pred:>6}  {conf:.2f}  {expected:>8}")
    if pred == expected:
        correct += 1

print("=" * 65)
print(f"\nAccuracy on test questions: {correct}/{len(test_questions)}")
print()

# Interactive mode
print("\n── Interactive mode (type 'quit' to exit) ──")
while True:
    q = input("\nEnter a question: ").strip()
    if q.lower() in ("quit", "exit", "q"): break
    if not q: continue
    pred, conf, all_probs = predict_difficulty(q)
    print(f"  Difficulty : {pred.upper()}  (confidence: {conf:.2%})")
    print(f"  All probs  : { {k: f'{v:.3f}' for k,v in all_probs.items()} }")