import numpy as np
import matplotlib.pyplot as plt

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([1.8, 6.0, 6.2, 10.5, 9.8, 15.1, 13.5, 18.9, 17.5, 23.2], dtype=float)

# Build design matrix [x, 1] for y = ax + b
A = np.column_stack([x, np.ones_like(x)])

# Solve least squares
coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
a, b = coeffs
print(f"slope = {a:.4f}, intercept = {b:.4f}")

x_fit = np.linspace(0.5, 10.5, 100)
y_fit = a * x_fit + b

fig, ax = plt.subplots(figsize=(5, 3.5))
ax.scatter(x, y, color="steelblue", zorder=3, label="Data")
ax.plot(x_fit, y_fit, color="tomato", linewidth=2,
        label=f"Fit: $y = {a:.2f}x + {b:.2f}$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("least_squares_fit.pdf", bbox_inches="tight")
plt.savefig("least_squares_fit.png", dpi=150, bbox_inches="tight")
print("Saved least_squares_fit.pdf and .png")
