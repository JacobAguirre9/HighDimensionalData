# This is code for the swiss roll implementation of LLE
# All figures taken were used in the 3D plot, so they aren't exactly reproducible.
# You can, however, move the plot around if you run the code in python and can find the same image dimensions
# citation of paper where I wrote down the algorithm and most of the methodology

# T. Cox and M. A. A. Cox, Multidimensional scaling. Boca Raton, FL, USA:
# CRC Press, 2000.

# Roweis, Sam T., and Lawrence K. Saul. "Nonlinear dimensionality reduction by locally linear embedding." science 290.5500 (2000): 2323-2326.

# Zhang, Z. & Wang, J. MLLE: Modified Locally Linear Embedding Using Multiple Weights.

# When looking at codes, these seemed to be every package I needed to use
import numpy as np
from scipy.linalg import eigh, svd, qr, solve
from ..utils import check_random_state, check_array
from ..utils._arpack import _init_arpack_v0
from ..utils.extmath import stable_cumsum
from ..base import (
    BaseEstimator,
    TransformerMixin,
    _UnstableArchMixin,
    _ClassNamePrefixFeaturesOutMixin,
)

from ..utils.validation import check_is_fitted
from ..utils.validation import FLOAT_DTYPES
from ..neighbors import NearestNeighbors
from scipy.sparse import eye, csr_matrix
from scipy.sparse.linalg import eigsh

# This function needs to take in a dataset X (here, a swiss roll, where I used the code from the other Isomap function to generate a swiss roll)
# X and Y both share the same format such that X,Y=(Number of Samples, number of dimensions)
def BCWeights(X, Y, indices, reg=1e-3):

    for i, ind in enumerate(indices):
        A = Y[ind]
        C = A - X[i] # this code is trying to simply compute the weights in each point in Y to recover the point X_i, its important to note this all must sum to 1
        G = np.dot(C, C.T) #dot product
        MatrixTrace = np.trace(G) # simply the trace of this matrix
        if MatrixTrace > 0:
            Regular = reg * trace
        else:
            Regular = reg
        G.flat[:: n_neighbors + 1] += Regular
        Wequation = solve(G, v, sym_pos=True)
        BMatrix[i, :] = w / np.sum(Wequation)
    return BMatrix


def BCKNeighborGraph(X, number_neighbors, reg=1e-3, n_jobs=None):
    KNeighestNeighbor = NearestNeighbors(number_neighbors=number_neighbors + 1, n_jobs=n_jobs).fit(X)
    X = KNeighestNeighbor._fit_X
    NumberOfSamples = KNeighestNeighbor.NumberOfSamples_fit_
    ind = KNeighestNeighbor.kneighbors(X, return_distance=False)[:, 1:]
    data = barycenter_weights(X, X, ind, reg=reg)
    indptr = np.arange(0, NumberOfSamples * number_neighbors + 1, number_neighbors)
    return csr_matrix((data.ravel(), ind.ravel(), indptr), shape=(NumberOfSamples, NumberOfSamples))


# Now, this follows standard methodology (??) it would seem that scikit follows!
# I think I may have had an error with the eigen solver, and max iterations computation

def locally_linear_embedding(
    X,
    *,
    n_neighbors,
    n_components,
    reg=1e-3,
    eigen_solver="auto",
    tol=1e-6,
    max_iter=100,
    method="standard",
    hessian_tol=1e-4,
    modified_tol=1e-12,
    random_state=None,
    n_jobs=None,
):
    if eigen_solver not in ("auto", "arpack", "dense"):
        raise ValueError("unrecognized eigen_solver '%s'" % eigen_solver)

    if method not in ("standard", "hessian", "modified", "ltsa"):
        raise ValueError("unrecognized method '%s'" % method)

    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, n_jobs=n_jobs)
    nbrs.fit(X)
    X = nbrs._fit_X

    N, d_in = X.shape

    if n_components > d_in:
        raise ValueError(
            "output dimension must be less than or equal to input dimension"
        )
    if n_neighbors >= N:
        raise ValueError(
            "Expected n_neighbors <= n_samples,  but n_samples = %d, n_neighbors = %d"
            % (N, n_neighbors)
        )

    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    M_sparse = eigen_solver != "dense"

    if method == "standard":
        W = barycenter_kneighbors_graph(
            nbrs, n_neighbors=n_neighbors, reg=reg, n_jobs=n_jobs
        )

        # we'll compute M = (I-W)'(I-W)
        # depending on the solver, we'll do this differently
        if M_sparse:
            M = eye(*W.shape, format=W.format) - W
            M = (M.T * M).tocsr()
        else:
            M = (W.T * W - W.T - W).toarray()
            M.flat[:: M.shape[0] + 1] += 1  # W = W - I = W - I

    elif method == "hessian":
        dp = n_components * (n_components + 1) // 2

        if n_neighbors <= n_components + dp:
            raise ValueError(
                "for method='hessian', n_neighbors must be "
                "greater than "
                "[n_components * (n_components + 3) / 2]"
            )

        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        neighbors = neighbors[:, 1:]

        Yi = np.empty((n_neighbors, 1 + n_components + dp), dtype=np.float64)
        Yi[:, 0] = 1

        M = np.zeros((N, N), dtype=np.float64)

        use_svd = n_neighbors > d_in

        for i in range(N):
            Gi = X[neighbors[i]]
            Gi -= Gi.mean(0)

            # build Hessian estimator
            if use_svd:
                U = svd(Gi, full_matrices=0)[0]
            else:
                Ci = np.dot(Gi, Gi.T)
                U = eigh(Ci)[1][:, ::-1]

            Yi[:, 1: 1 + n_components] = U[:, :n_components]

            j = 1 + n_components
            for k in range(n_components):
                Yi[:, j: j + n_components - k] = U[:, k: k + 1] * U[:, k:n_components]
                j += n_components - k

            Q, R = qr(Yi)

            w = Q[:, n_components + 1:]
            S = w.sum(0)

            S[np.where(abs(S) < hessian_tol)] = 1
            w /= S

            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(w, w.T)

        # we sort of have to add in this clause, simply because many of the matrices we'll be dealing with are sparse
        if M_sparse:
            M = csr_matrix(M)

        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        neighbors = neighbors[:, 1:]


        V = np.zeros((N, n_neighbors, n_neighbors))
        nev = min(d_in, n_neighbors)
        evals = np.zeros([N, nev])

        # choose the most efficient way to find the eigenvectors
        use_svd = n_neighbors > d_in

        if use_svd:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                V[i], evals[i], _ = svd(X_nbrs, full_matrices=True)
            evals **= 2
        else:
            for i in range(N):
                X_nbrs = X[neighbors[i]] - X[i]
                C_nbrs = np.dot(X_nbrs, X_nbrs.T)
                evi, vi = eigh(C_nbrs)
                evals[i] = evi[::-1]
                V[i] = vi[:, ::-1]

        reg = 1e-3 * evals.sum(1)

        tmp = np.dot(V.transpose(0, 2, 1), np.ones(n_neighbors))
        tmp[:, :nev] /= evals + reg[:, None]
        tmp[:, nev:] /= reg[:, None]

        w_reg = np.zeros((N, n_neighbors))
        for i in range(N):
            w_reg[i] = np.dot(V[i], tmp[i])
        w_reg /= w_reg.sum(1)[:, None]

        # What does \eta mean? from the literature this is simply the median of a
        # ratio of the smaller to larger eigenvalues
        # across all the points.  This is used to determine s_i
        # so, we have that :
        rho = evals[:, n_components:].sum(1) / evals[:, :n_components].sum(1)
        eta = np.median(rho)


        s_range = np.zeros(N, dtype=int)
        evals_cumsum = stable_cumsum(evals, 1)
        eta_range = evals_cumsum[:, -1:] / evals_cumsum[:, :-1] - 1
        for i in range(N):
            s_range[i] = np.searchsorted(eta_range[i, ::-1], eta)
        s_range += n_neighbors - nev  # number of zero eigenvalues

        # Now, we define M to be a N x N matrix (square, invertible and allows us to have nice properties)
        M = np.zeros((N, N), dtype=np.float64)
        for i in range(N):
            s_i = s_range[i]

            # we must choose the lowest vectors and we're trying to compute \alpha
            Vi = V[i, :, n_neighbors - s_i:]
            alpha_i = np.linalg.norm(Vi.sum(0)) / np.sqrt(s_i)

                #h is simply the householder matrix, as given by the paper
            h = np.full(s_i, alpha_i) - np.dot(Vi.T, np.ones(n_neighbors))

            norm_h = np.linalg.norm(h)
            if norm_h < modified_tol:
                h *= 0
            else:
                h /= norm_h


            Wi = Vi - 2 * np.outer(np.dot(Vi, h), h) + (1 - alpha_i) * w_reg[i, :, None]

          # we now have to update this initial matrix
            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] += np.dot(Wi, Wi.T)
            Wi_sum1 = Wi.sum(1)
            M[i, neighbors[i]] -= Wi_sum1
            M[neighbors[i], i] -= Wi_sum1
            M[i, i] += s_i

        if M_sparse:
            M = csr_matrix(M)

    elif method == "ltsa":
        neighbors = nbrs.kneighbors(
            X, n_neighbors=n_neighbors + 1, return_distance=False
        )
        neighbors = neighbors[:, 1:]

        M = np.zeros((N, N))

        use_svd = n_neighbors > d_in

        for i in range(N):
            Xi = X[neighbors[i]]
            Xi -= Xi.mean(0)

            # compute n_components largest eigenvalues of Xi * Xi^T
            if use_svd:
                v = svd(Xi, full_matrices=True)[0]
            else:
                Ci = np.dot(Xi, Xi.T)
                v = eigh(Ci)[1][:, ::-1]

            Gi = np.zeros((n_neighbors, n_components + 1))
            Gi[:, 1:] = v[:, :n_components]
            Gi[:, 0] = 1.0 / np.sqrt(n_neighbors)

            GiGiT = np.dot(Gi, Gi.T)

            nbrs_x, nbrs_y = np.meshgrid(neighbors[i], neighbors[i])
            M[nbrs_x, nbrs_y] -= GiGiT
            M[neighbors[i], neighbors[i]] += 1

    return null_space(
        M,
        n_components,
        k_skip=1,
        eigen_solver=eigen_solver,
        tol=tol,
        max_iter=max_iter,
        random_state=random_state,
    )


class LocallyLinearEmbedding(
    _ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _UnstableArchMixin,
    BaseEstimator,
):
    # this is some code to compute the embedding vectors for a given data matrix, X

    def fitofMatrix(self, X, y=None):
        self._fit_transform(X)
        return self
    # Now, we need to transform these points into a new embedding space
    # so, take the training set of data X and this code will return a new matrix X
    # in the form again of (number of samples, number of dimensions)
    def transform(self, DataMatrixX):
        check_is_fitted(self)

        DataMatrixX = self._validate_data(DataMatrixX, reset=False)
        ind = self.nbrs_.kneighbors(
            DataMatrixX, n_neighbors=self.n_neighbors, return_distance=False
        )
        weights = BCWeights(DataMatrixX, self.nbrs_._fit_X, ind, reg=self.reg)
        X_newmatrix = np.empty((DataMatrixX.shape[0], self.n_components))
        for i in range(DataMatrixX.shape[0]):
            X_newmatrix[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
        return X_newmatrix



    #Note that we had to use the BCweights function we defined earlier in the above code
    # it's quite nice to see how all of this fits together
    # For the examples of Swiss roll, simply take the code from swiss roll visual py, insert here
    # again, I don't really know how to run code by sections (which is very annoying) so I didn't include the code on here for the final section
    # taking the time to run this code takes quite a bit, and I am not as proud of this work as I am with Isomap's code.
    # Thank you