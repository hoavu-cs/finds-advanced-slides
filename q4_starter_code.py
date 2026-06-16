# Install PyTorch (CPU-only):
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
#
# For GPU support, see https://pytorch.org/get-started/locally/

import torch
import csv

# load data.csv
rows = list(csv.DictReader(open('data.csv')))
x1 = torch.tensor([float(r['x1']) for r in rows])
x2 = torch.tensor([float(r['x2']) for r in rows])
y  = torch.tensor([float(r['y'])  for r in rows])

# parameters: a, b, c, d, e
params = torch.zeros(5, requires_grad=True)
opt    = torch.optim.SGD([params], lr=1e-3)

# Fill in the 4 loop
# for iter in range(10_000):


print(f"a={params[0]:.3f}  b={params[1]:.3f}  c={params[2]:.3f}")
print(f"d={params[3]:.3f}  e={params[4]:.3f}")
print(f"loss={loss.item():.3f}")
# true: a=3, b=-2, c=0.5, d=1, e=20
