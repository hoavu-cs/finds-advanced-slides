import torch
import numpy as np
import matplotlib.pyplot as plt

X = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.float)
y = torch.tensor([38, 47, 53, 58, 63, 60, 62], dtype=torch.float)

theta = torch.zeros(3, requires_grad=True)
opt   = torch.optim.SGD([theta], lr=1e-3)

for _ in range(50_000):
    y_hat = theta[0]*X**2 + theta[1]*X + theta[2]
    loss  = ((y - y_hat)**2).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()

print(theta.detach())  # ~ [-0.25,  8.25, 33.86]

a, b, c = theta.detach().numpy()
x_plot = np.linspace(0.5, 7.5, 200)
y_plot = a*x_plot**2 + b*x_plot + c

plt.figure(figsize=(7, 4))
plt.scatter(X.numpy(), y.numpy(), color="tab:orange", zorder=3, label="Data")
plt.plot(x_plot, y_plot, color="teal",
         label=f"Fit: {a:.2f}x² + {b:.2f}x + {c:.2f}")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Quadratic Fit (PyTorch)")
plt.legend()
plt.tight_layout()
plt.savefig("fit_torch.png", dpi=150)
plt.show()
