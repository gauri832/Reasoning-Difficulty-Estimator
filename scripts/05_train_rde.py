# scripts/05_train_rde.py
# Phase 2: Train the Reasoning Difficulty Estimator (RDE)
# Input:  data/features/labeled_signals.csv
# Output: models/rde/best_model.pt + training metrics

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import json

os.makedirs("models/rde", exist_ok=True)
os.makedirs("data/features", exist_ok=True)

# ── Config ────────────────────────────────────────────────
FEATURES    = ["attn_entropy", "varentropy", "perplexity",
               "tree_depth", "clause_count", "avg_sent_len"]
HIDDEN_DIMS = [256, 128, 64]
DROPOUT     = 0.3
LR          = 1e-3
BATCH_SIZE  = 32
EPOCHS      = 60
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
SEED        = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── Load signals + embeddings ────────────────────────────
import json

df = pd.read_csv("data/features/labeled_signals.csv")

with open("data/features/signals.json") as f:
    signals_full = json.load(f)

# id → embedding map
emb_map = {item["id"]: item["embedding"] for item in signals_full if "embedding" in item}

# keep only rows with embeddings
df = df[df["id"].isin(emb_map)].reset_index(drop=True)

# ── Build features ───────────────────────────────────────
SCALAR_COLS = ["attn_entropy", "varentropy", "perplexity",
               "tree_depth", "clause_count", "avg_sent_len"]

scalar_feats = df[SCALAR_COLS].fillna(0).values.astype(np.float32)

emb_feats = np.array(
    [emb_map[id_] for id_ in df["id"]],
    dtype=np.float32
)

# Normalize scalars
scalar_mean = scalar_feats.mean(axis=0)
scalar_std  = scalar_feats.std(axis=0) + 1e-8
scalar_norm = (scalar_feats - scalar_mean) / scalar_std

# Normalize embeddings
emb_mean = emb_feats.mean(axis=0)
emb_std  = emb_feats.std(axis=0) + 1e-8
emb_norm = (emb_feats - emb_mean) / emb_std

# Combine
X = np.concatenate([scalar_norm, emb_norm], axis=1)

# ── Labels ───────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(["easy", "medium", "hard"])   # correct order
y = le.transform(df["difficulty"])

print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
print("Class counts:\n", df["difficulty"].value_counts())

# Save label classes
np.save("models/rde/label_classes.npy", le.classes_)

# Save scalers
np.save("models/rde/scalar_mean.npy", scalar_mean)
np.save("models/rde/scalar_scale.npy", scalar_std)
np.save("models/rde/emb_mean.npy", emb_mean)
np.save("models/rde/emb_scale.npy", emb_std)

# ── Train / Val / Test split (70/15/15) ───────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=SEED) 
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")


# ── Dataset ───────────────────────────────────────────────
class SignalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_loader = DataLoader(SignalDataset(X_train, y_train),
                          batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SignalDataset(X_val,   y_val),
                          batch_size=BATCH_SIZE)
test_loader  = DataLoader(SignalDataset(X_test,  y_test),
                          batch_size=BATCH_SIZE)


# ── RDE Model (MLP with BatchNorm + Dropout) ──────────────
class RDEClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def predict_proba(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=-1)

input_dim= X.shape[1]
model = RDEClassifier(
    input_dim   = input_dim,
    hidden_dims = HIDDEN_DIMS,
    num_classes = len(le.classes_),
    dropout     = DROPOUT
).to(DEVICE)

print(f"\nRDE Model:\n{model}\n")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Class weights to handle imbalance ─────────────────────
class_counts = np.bincount(y_train)
class_weights = torch.tensor(
    1.0 / class_counts, dtype=torch.float32
).to(DEVICE)
class_weights = class_weights / class_weights.sum() * len(le.classes_)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", patience=8, factor=0.5)

# ── Training loop ─────────────────────────────────────────
def evaluate(loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            logits = model(X_b)
            loss   = criterion(logits, y_b)
            preds  = logits.argmax(dim=1)
            correct   += (preds == y_b).sum().item()
            total     += len(y_b)
            loss_sum  += loss.item() * len(y_b)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_b.cpu().numpy())
    return correct / total, loss_sum / total, all_preds, all_labels


best_val_acc   = 0.0
history        = {"train_loss": [], "val_loss": [], "val_acc": []}
patience_count = 0
EARLY_STOP     = 15

print("=" * 55)
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  {'Val Acc':>8}")
print("=" * 55)

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y_b)
    train_loss /= len(X_train)

    val_acc, val_loss, _, _ = evaluate(val_loader)
    scheduler.step(val_acc)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if epoch % 5 == 0 or epoch == 1:
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}  {val_acc:>8.4f}")

    if val_acc > best_val_acc:
        best_val_acc   = val_acc
        patience_count = 0
        torch.save({
            "model_state":  model.state_dict(),
            "input_dim":    len(FEATURES),
            "hidden_dims":  HIDDEN_DIMS,
            "num_classes":  len(le.classes_),
            "dropout":      DROPOUT,
            "features":     FEATURES,
            "label_classes":list(le.classes_),
            "val_acc":      val_acc
        }, "models/rde/best_model.pt")
    else:
        patience_count += 1
        if patience_count >= EARLY_STOP:
            print(f"\nEarly stopping at epoch {epoch}")
            break

print(f"\n✅ Best Val Accuracy: {best_val_acc:.4f}")

# Save training history
with open("models/rde/training_history.json", "w") as f:
    json.dump(history, f, indent=2)


# ── Final test evaluation ─────────────────────────────────
checkpoint = torch.load(
    "models/rde/best_model.pt",
    map_location=DEVICE,
    weights_only=False   # 🔥 FIX
)
model.load_state_dict(checkpoint["model_state"])

test_acc, test_loss, preds, labels = evaluate(test_loader)

print("\n" + "=" * 55)
print("FINAL TEST RESULTS")
print("=" * 55)
print(f"Test Accuracy : {test_acc:.4f}")
print(f"Test Loss     : {test_loss:.4f}")
print()
print("Per-class Report:")
print(classification_report(labels, preds,
      target_names=le.classes_, digits=3))

print("Confusion Matrix (rows=actual, cols=predicted):")
cm = confusion_matrix(labels, preds)
cm_df = pd.DataFrame(cm,
    index=[f"actual_{c}"    for c in le.classes_],
    columns=[f"pred_{c}"    for c in le.classes_])
print(cm_df.to_string())

# Save test results for paper
results = {
    "test_accuracy": test_acc,
    "test_loss":     test_loss,
    "best_val_acc":  best_val_acc,
    "per_class":     classification_report(
                        labels, preds,
                        target_names=le.classes_,
                        output_dict=True)
}
with open("models/rde/test_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Model saved to models/rde/best_model.pt")
print(f"✅ Results saved to models/rde/test_results.json")