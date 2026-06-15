import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- data ---
X, y = load_digits(return_X_y=True)
X = X.reshape(-1, 1, 8, 8) / 16.0    # shape: (n, 1, 8, 8)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_val   = torch.tensor(X_val,   dtype=torch.float32)
y_val   = torch.tensor(y_val,   dtype=torch.long)

# --- baseline CNN ---
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),  # -> 8 x 8 x 8
    nn.ReLU(),
    nn.MaxPool2d(2),                             # -> 8 x 4 x 4
    nn.Flatten(),
    nn.Linear(8 * 4 * 4, 10),
)

loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

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

print(f"Baseline validation accuracy: {val_accs[-1]:.3f}")

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
