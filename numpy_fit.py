import numpy as np
import matplotlib.pyplot as plt

X = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)
y = np.array([38, 47, 53, 58, 63, 60, 62], dtype=float)

theta = np.zeros(3)  # [a, b, c]
alpha = 1e-3

for _ in range(50_000):
    y_hat = theta[0]*X**2 + theta[1]*X + theta[2]
    err   = y_hat - y
    grad  = np.array([2*(err*X**2).mean(),
                      2*(err*X).mean(),
                      2*err.mean()])
    theta -= alpha * grad

print(theta)  # ~ [-0.25,  8.25, 33.86]

a, b, c = theta
x_plot = np.linspace(0.5, 7.5, 200)
y_plot = a*x_plot**2 + b*x_plot + c

plt.figure(figsize=(7, 4))
plt.scatter(X, y, color="tab:orange", zorder=3, label="Data")
plt.plot(x_plot, y_plot, color="teal",
         label=f"Fit: {a:.2f}x² + {b:.2f}x + {c:.2f}")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Quadratic Fit (NumPy)")
plt.legend()
plt.tight_layout()
plt.savefig("fit_numpy.png", dpi=150)
plt.show()
