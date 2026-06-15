import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- data ---
X, y = load_digits(return_X_y=True)
X = X / 16.0

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.long)

# --- model (TODO: fill in layers) ---
model = nn.Sequential(
    # your layers here
    # input size: 64  (8x8 flattened image)
    # output size: 10 (one logit per digit class)
)

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# --- training loop ---
train_losses, val_accs = [], []

for epoch in range(200):
    model.train()
    logits = model(X_train)
    loss   = loss_fn(logits, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        preds = model(X_val).argmax(dim=1)
        acc   = (preds == y_val).float().mean().item()
    val_accs.append(acc)

print(f"Final validation accuracy: {val_accs[-1]:.3f}")

# --- plots ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(train_losses)
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax1.set_title("Training Loss")
ax2.plot(val_accs)
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy")
ax2.set_title("Validation Accuracy")
plt.tight_layout()
plt.show()
