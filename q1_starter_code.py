# Problem 1: Cubic Fitting with PyTorch
# Install dependencies: pip install torch matplotlib --index-url https://download.pytorch.org/whl/cpu

import torch
import matplotlib.pyplot as plt

# Data
t = torch.tensor([-3.00, -2.57, -2.14, -1.71, -1.29, -0.86, -0.43,  0.00,
                    0.43,  0.86,  1.29,  1.71,  2.14,  2.57,  3.00], dtype=torch.float)
y = torch.tensor([-76.39, -55.41, -34.20, -21.28, -18.66, -12.17, -6.37, -5.63,
                   -3.77,  -3.61,   3.21,   8.84,  16.65,  32.36, 49.94], dtype=torch.float)

# Parameters (a, b, c, d) 
a = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
c = torch.tensor(0.0, requires_grad=True)
d = torch.tensor(0.0, requires_grad=True)

# Optimizer 
# ..............


# Write the training training loop 
# ..............

# Result
print(f"\na = {a.item():.3f}")
print(f"b = {b.item():.3f}")
print(f"c = {c.item():.3f}")
print(f"d = {d.item():.3f}")
print(f"final loss = {loss.item():.4f}")

# Plot
t_smooth = torch.linspace(-3, 3, 200)
with torch.no_grad():
    y_smooth = a*t_smooth**3 + b*t_smooth**2 + c*t_smooth + d

plt.figure()
plt.scatter(t.numpy(), y.numpy(), color='black', zorder=5, label='data')
plt.plot(t_smooth.numpy(), y_smooth.numpy(), color='blue', label='fitted curve')
plt.xlabel('t')
plt.ylabel('y')
plt.title('Cubic Fitting')
plt.legend()
plt.tight_layout()
plt.show()
