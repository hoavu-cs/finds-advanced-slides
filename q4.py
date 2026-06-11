import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.00, 0.42, 0.84, 1.26, 1.68, 2.11, 2.53, 2.95,
              3.37, 3.79, 4.21, 4.63, 5.05, 5.47, 5.89, 6.32,
              6.74, 7.16, 7.58, 8.00])
y = np.array([25.07, 14.75, 13.11, 11.80,  6.48,  7.81,  7.50,  2.64,
              12.07, 12.63, 11.47, 16.06, 22.02, 24.36, 29.76, 32.33,
              45.12, 51.30, 59.94, 63.42])

# (a) Build design matrix with columns [x^2, x, 1]
# TODO: A = np.column_stack([x**2, x, np.ones_like(x)])

# (b) Solve least squares and report a, b, c
# TODO: coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
# TODO: a, b, c = coeffs
# TODO: print(f"y = {a:.4f} x^2 + {b:.4f} x + {c:.4f}")

# (c) Plot data and fitted curve
# TODO: x_fit = np.linspace(0, 8, 200)
# TODO: y_fit = a * x_fit**2 + b * x_fit + c
# TODO: plt.scatter(x, y, label="Data")
# TODO: plt.plot(x_fit, y_fit, label=f"Fit: {a:.2f}x² + {b:.2f}x + {c:.2f}")
# TODO: plt.legend(); plt.xlabel("x"); plt.ylabel("y"); plt.show()

# (d) Compute RMSE
# TODO: residuals = A @ coeffs - y
# TODO: rmse = np.sqrt(np.mean(residuals**2))
# TODO: print(f"RMSE: {rmse:.4f}")
