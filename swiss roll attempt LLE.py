# This is code for the swiss roll implementation of LLE
# All figures taken were used in the 3D plot, so they aren't exactly reproducible.
# You can, however, move the plot around if you run the code in python and can find the same image dimensions
# citation of paper where I wrote down the algorithm and most of the methodology
# T. Cox and M. A. A. Cox, Multidimensional scaling. Boca Raton, FL, USA:
# CRC Press, 2000.

# Roweis, Sam T., and Lawrence K. Saul. "Nonlinear dimensionality reduction by locally linear embedding." science 290.5500 (2000): 2323-2326.

# Zhang, Z. & Wang, J. MLLE: Modified Locally Linear Embedding Using Multiple Weights.
import matplotlib
import numpy as np
import scipy

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets, neighbors

# Swiss roll dataset
NumberOfSamples = 3000
X, color = datasets.samples_generator.make_swiss_roll(n_samples = NumberOfSamples, noise = 0.5)

#notice how the noise is quite high, as we could tell from the HW when noise is over .2, spectral graph clustering gets quite ahrd
# 3D version
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c = color, cmap = plt.cm.Spectral)

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z');


# Visualize dataset in 2D, projecting into the xz-plane
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(X[:, 0], X[:, 2], c = color, cmap = plt.cm.Spectral)

ax.set_title('2D  Swiss roll')
ax.set_xlabel('x')
ax.set_ylabel('z')

# Compute k nearest neighbors to each point
k = 24
nbors = neighbors.kneighbors_graph(X, n_neighbors = k)

# Compute reconstruction weights
W = np.zeros((NumberOfSamples, NumberOfSamples))
for i, nbor_row in enumerate(nbors):
    # Get the indices of the nonzero entries in each row of the
    # neighbors matrix (which is sparse). These are the nearest
    # neighbors to the point in question. dim(Z) = [K, D]
    inds = nbor_row.indices
    Z = X[inds] - X[i]

    # Local covariance. Regularize because our data is
    # low-dimensional (K > D). dim(C) = [K, K]
    C = Z @ Z.T
    C += np.eye(k) * np.trace(C) * 0.001

    # Compute reconstruction weights
    w = scipy.linalg.solve(C, np.ones(k))
    W[i, inds] = w / sum(w)

    # Create sparse, symmetric matrix
    M = (np.eye(NumberOfSamples) - W).T @ (np.eye(NumberOfSamples) - W)
    M = M.T

    # Find bottom d+1 eigenvectors
    d = 2
    vals, vecs = scipy.linalg.eigh(M, eigvals=(0, d))

    X2 = vecs[:, 1:]
    plt.scatter(X2[:, 0], X2[:, 1], c=color, cmap=plt.cm.Spectral)