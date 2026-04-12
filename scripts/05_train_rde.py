import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

os.makedirs("models/rde", exist_ok=True)

SCALAR_FEATURES = [
    "attn_entropy",
    "varentropy",
    "perplexity",
    "tree_depth",
    "clause_count",
    "avg_sent_len",
    "proof_kw_density",
    "math_sym_density",
    "eq_density",
    "abstract_ratio",
    "avg_word_len",
    "num_density",
]

CLASS_NAMES = ["easy", "medium", "hard"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

HIDDEN_DIMS = [256, 128, 64]
DROPOUT = 0.30
LR = 1e-3
BATCH_SIZE = 32
EPOCHS = 90
STOP = 18
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


def safe_class_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = len(y) / (n_classes * counts)
    return weights.astype(np.float32)


class SignalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class RDEClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims, num_classes: int, dropout: float):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evaluate(model, loader, criterion):
    model.eval()
    preds_all = []
    labels_all = []
    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)

            n = len(yb)
            total += n
            correct += (preds == yb).sum().item()
            loss_sum += loss.item() * n
            preds_all.extend(preds.cpu().numpy().tolist())
            labels_all.extend(yb.cpu().numpy().tolist())

    acc = correct / max(total, 1)
    avg_loss = loss_sum / max(total, 1)
    macro_f1 = f1_score(labels_all, preds_all, average="macro", zero_division=0)
    return acc, avg_loss, macro_f1, preds_all, labels_all


def main() -> None:
    df = pd.read_csv("data/features/labeled_signals_v2.csv")
    df = df.dropna(subset=SCALAR_FEATURES + ["difficulty"])
    df = df[df["difficulty"].isin(CLASS_NAMES)].copy()

    with open("data/features/signals_v2.json", "r", encoding="utf-8") as f:
        signals = json.load(f)
    emb_map = {item["id"]: item["embedding"] for item in signals if "embedding" in item}

    df = df[df["id"].isin(emb_map)].reset_index(drop=True)

    print(f"Dataset: {len(df)} samples")
    print("Class distribution:\n", df["difficulty"].value_counts(), "\n")

    scalar_feats = df[SCALAR_FEATURES].fillna(0.0).values.astype(np.float32)
    emb_feats = np.array([emb_map[i] for i in df["id"]], dtype=np.float32)

    sc_mean = scalar_feats.mean(axis=0)
    sc_std = scalar_feats.std(axis=0) + 1e-8
    em_mean = emb_feats.mean(axis=0)
    em_std = emb_feats.std(axis=0) + 1e-8

    scalar_norm = (scalar_feats - sc_mean) / sc_std
    emb_norm = (emb_feats - em_mean) / em_std
    x = np.concatenate([scalar_norm, emb_norm], axis=1).astype(np.float32)

    np.save("models/rde/scalar_mean.npy", sc_mean)
    np.save("models/rde/scalar_scale.npy", sc_std)
    np.save("models/rde/emb_mean.npy", em_mean)
    np.save("models/rde/emb_scale.npy", em_std)
    np.save("models/rde/label_classes.npy", np.array(CLASS_NAMES))

    y = df["difficulty"].map(CLASS_TO_IDX).astype(np.int64).values
    print("Class mapping:", CLASS_TO_IDX)

    x_tr, x_tmp, y_tr, y_tmp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=SEED
    )
    x_va, x_te, y_va, y_te = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )
    print(f"Train: {len(x_tr)} | Val: {len(x_va)} | Test: {len(x_te)}")

    tr_ds = SignalDataset(x_tr, y_tr)
    va_ds = SignalDataset(x_va, y_va)
    te_ds = SignalDataset(x_te, y_te)

    counts = np.bincount(y_tr, minlength=len(CLASS_NAMES))
    sample_weights = 1.0 / np.maximum(counts[y_tr], 1)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(sample_weights),
        replacement=True,
    )

    tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, sampler=sampler)
    va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE)
    te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE)

    model = RDEClassifier(
        input_dim=x.shape[1],
        hidden_dims=HIDDEN_DIMS,
        num_classes=len(CLASS_NAMES),
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    class_weights = torch.tensor(
        safe_class_weights(y_tr, len(CLASS_NAMES)), dtype=torch.float32, device=DEVICE
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=6
    )

    best_val_f1 = -1.0
    patience = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}

    print(f"{'Epoch':>6}  {'TrLoss':>8}  {'VaLoss':>8}  {'VaAcc':>7}  {'VaF1':>7}")
    print("-" * 52)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_loss = 0.0

        for xb, yb in tr_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            tr_loss += loss.item() * len(yb)

        tr_loss /= max(len(x_tr), 1)

        va_acc, va_loss, va_f1, _, _ = evaluate(model, va_loader, criterion)
        scheduler.step(va_f1)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_macro_f1"].append(va_f1)

        if epoch == 1 or epoch % 5 == 0:
            print(f"{epoch:>6}  {tr_loss:>8.4f}  {va_loss:>8.4f}  {va_acc:>7.4f}  {va_f1:>7.4f}")

        improved = va_f1 > best_val_f1 + 1e-5
        if improved:
            best_val_f1 = va_f1
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_dim": int(x.shape[1]),
                    "hidden_dims": HIDDEN_DIMS,
                    "num_classes": len(CLASS_NAMES),
                    "dropout": DROPOUT,
                    "scalar_features": SCALAR_FEATURES,
                    "label_classes": CLASS_NAMES,
                    "val_acc": float(va_acc),
                    "val_macro_f1": float(va_f1),
                },
                "models/rde/best_model.pt",
            )
        else:
            patience += 1
            if patience >= STOP:
                print(f"\nEarly stop @ epoch {epoch}")
                break

    print(f"\nBest Val Macro-F1: {best_val_f1:.4f}")

    with open("models/rde/training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load("models/rde/best_model.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    te_acc, te_loss, te_f1, preds, labels = evaluate(model, te_loader, criterion)

    print("\n" + "=" * 55)
    print("TEST RESULTS")
    print("=" * 55)
    print(f"Accuracy: {te_acc:.4f}  |  Loss: {te_loss:.4f}  |  Macro-F1: {te_f1:.4f}\n")

    print(
        classification_report(
            labels,
            preds,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            digits=3,
            zero_division=0,
        )
    )

    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASS_NAMES))))
    cm_df = pd.DataFrame(
        cm,
        index=[f"act_{c}" for c in CLASS_NAMES],
        columns=[f"pred_{c}" for c in CLASS_NAMES],
    )
    print("Confusion Matrix:")
    print(cm_df.to_string())

    results = {
        "test_accuracy": float(te_acc),
        "test_macro_f1": float(te_f1),
        "best_val_macro_f1": float(best_val_f1),
        "per_class": classification_report(
            labels,
            preds,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
    }
    with open("models/rde/test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved models/rde/best_model.pt")

    # First-layer absolute weights as a rough feature-importance proxy.
    first_linear = model.net[0]
    weights = first_linear.weight.detach().abs().mean(dim=0).cpu().numpy()
    scalar_imp = list(zip(SCALAR_FEATURES, weights[: len(SCALAR_FEATURES)]))
    scalar_imp.sort(key=lambda t: t[1], reverse=True)

    print("\nTop scalar feature importances:")
    for name, value in scalar_imp:
        bar = "#" * max(1, int(value * 45))
        print(f"  {name:<22} {value:.4f}  {bar}")


if __name__ == "__main__":
    main()
