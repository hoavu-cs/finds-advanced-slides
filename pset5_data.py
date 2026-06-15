import numpy as np

# ── Graph for Q1 / Q2 ──────────────────────────────────────────────────────
np.random.seed(42)
n = 10
# Random directed graph: each node has 2-4 random outgoing edges
edges = set()
for i in range(n):
    targets = np.random.choice([j for j in range(n) if j != i],
                               size=np.random.randint(2, 5), replace=False)
    for t in targets:
        edges.add((i, int(t)))

# Adjacency matrix (A[i][j] = 1 means edge i -> j)
A = np.zeros((n, n), dtype=int)
for (i, j) in sorted(edges):
    A[i][j] = 1

print("Edge list:")
for (i, j) in sorted(edges):
    print(f"  {i} -> {j}")

print("\nAdjacency matrix (row i = outgoing edges from node i):")
print(A)

# ── Quadratic data for Q4 ──────────────────────────────────────────────────
np.random.seed(7)
x_q4 = np.round(np.linspace(0, 8, 20), 2)
noise = np.random.randn(20) * 3
y_q4 = np.round(2 * x_q4**2 - 10 * x_q4 + 20 + noise, 2)

print("\nQ4 data (x, y):")
for xi, yi in zip(x_q4, y_q4):
    print(f"  x={xi:.2f}, y={yi:.2f}")

print("\nx array:", list(x_q4))
print("y array:", list(y_q4))
