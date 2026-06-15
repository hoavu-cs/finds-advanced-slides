import numpy as np

# Reuse the transition matrix M built in q1.py
# Copy your M here, or import it:
#   from q1 import M

A = np.array([
    [0,1,1,0,0,0,1,0,0,1],
    [1,0,1,0,0,0,1,0,0,1],
    [0,0,0,1,0,1,0,0,0,0],
    [0,1,0,0,0,0,0,1,1,0],
    [0,1,0,0,0,1,1,0,0,0],
    [0,0,1,0,0,0,1,0,1,1],
    [1,0,1,0,0,1,0,0,0,0],
    [0,1,0,1,1,0,0,0,0,1],
    [1,0,0,0,0,0,1,1,0,0],
    [0,1,1,0,1,0,0,0,0,0],
], dtype=float)

d = 0.85
n = 10

# Build M (same as q1)
# TODO

# (a) Power iteration starting from uniform distribution
r = np.ones(n) / n

# TODO: iterate r = d * M @ r + (1-d)/n * ones until convergence

# (b) Print number of iterations to converge

# (c) Compare with q1 result: print max absolute difference
