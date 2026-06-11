import numpy as np

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

# (a) Build the column-stochastic transition matrix M
#     M[j, i] = 1 / out_degree(i) if edge i -> j, else 0
# TODO

# (b) Set up and solve (I - d*M) r = (1-d)/n * ones
# TODO

# (c) Normalise r so it sums to 1, then print rankings
# TODO
