import numpy as np

# Download the dataset from:
# https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009
# Place winequality-red.csv in the same folder as this file.

# (a) Load the CSV and build design matrix A and target vector b
# TODO: load with np.genfromtxt or pandas
#   data = ...
#   features = data[:, :-1]   # 11 feature columns
#   quality  = data[:, -1]    # quality column

# Add a bias column of ones to get A shape (n, 12)
# TODO: A = np.column_stack([features, np.ones(len(features))])
# TODO: b = quality

# (b) Solve least squares
# TODO: coeffs, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
# TODO: print coeffs

# (c) Compute RMSE on training set
# TODO: residuals = A @ coeffs - b
# TODO: rmse = np.sqrt(np.mean(residuals**2))
# TODO: print(f"RMSE: {rmse:.4f}")

# (d) Which feature has the largest absolute coefficient?
feature_names = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"
]
# TODO: find and print the feature with the largest |coefficient|
