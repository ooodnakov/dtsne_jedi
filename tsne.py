#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import matplotlib.pyplot as plt
import argparse
import tqdm

def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta) 
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P

def BetaDens(D=np.array([]), beta=np.array([]), i=None):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    n = D.shape[0]
    #symm_beta = 1. / ((1. / np.sqrt(beta[np.concatenate((np.r_[0:i], np.r_[i+1:n+1]))] / 2.) + 1. / np.sqrt(beta[i] / 2.)) / 2.)**2 / 2.
    symm_beta = 4 * beta * beta[i] / (2 * np.sqrt(beta * beta[i]) + beta[i] + beta)
    gamma = symm_beta.copy()
    #print(beta[i], symm_beta[i], gamma[i], beta.mean(), symm_beta.mean(), gamma.mean())
    P = np.exp(-D.copy() * symm_beta[np.concatenate((np.r_[0:i], np.r_[i+1:n+1])),0])
    sumP = sum(P)
    P = P / sumP
    return P, gamma[np.concatenate((np.r_[0:i], np.r_[i+1:n+1])),0]

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0, dens=False):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    gamma = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])
        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    if dens:
        for i in range(n):
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
            thisP, thisgamma = BetaDens(Di, beta, i)
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
            gamma[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisgamma
        # plt.plot(np.convolve(np.maximum(gamma[:,:50],1e-12).mean(0), np.ones(50), 'valid') / 50)
        # plt.show()
        #print(gamma, beta)
        gamma /= gamma.max()
    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P, gamma


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to get initiall embedding.
    """

    print("Getting initall embedding using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, perplexity=30.0, max_iter=1000, dens=False):
    np.random.seed(1)
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    #X = pca(X, 50).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = pca(X, no_dims).real
    Y = 1e-4 * Y / np.std(Y)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P, gamma = x2p(X, 1e-5, perplexity, dens)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # early exaggeration
    P = P * 4.  
    P = np.maximum(P, np.finfo(np.double).eps)
    C_old = np.sum(P)+100
    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        if dens:
            num = 1. / (1. + gamma * np.add(np.add(num, sum_Y).T, sum_Y))
        else:
            num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        if dens:
            for i in range(n):
                dY[i, :] = 4 * np.sum(np.tile(PQ[:, i] * num[:, i] * gamma[:, i], (no_dims, 1)).T * (Y[i, :] - Y) , 0)
        else:
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        #Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            if abs(C-C_old)<5e-5 and iter > 100:
                print("Early stopping due to small change", C, C_old)
                break
            C_old = C
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    
    # plt.figure(figsize=(10,10))
    # plt.imshow(Q, cmap='hot', interpolation='nearest')
    # plt.show()
    # plt.figure(figsize=(10,10))
    # plt.imshow(dY[:,:], cmap='hot', interpolation='nearest')
    # plt.show()
    # Return solution
    return Y


def get_gaussian_data(dims=2, n_clusters=5):
    X = []
    y = []
    from scipy.stats import ortho_group

    for cluster in range(n_clusters):
        means = 100 * (np.random.rand(dims) - 0.5)
        diag = np.abs(np.diag(40 * np.random.rand(dims)))
        O = ortho_group.rvs(dims)
        covs = O.T @ diag @ O
        num_samples = np.random.randint(10, 200)

        cloud = np.random.multivariate_normal(means, covs, num_samples)

        X.append(cloud)
        y.append([cluster] * num_samples)

    return np.concatenate(X), np.concatenate(y)
    
if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--dens", type=int, default=0, help="use dtSNE")
    parser.add_argument("--nDims", type=int, default=2, help="nDims")
    parser.add_argument("--iter", type=int, default=300, help="iterations")
    parser.add_argument("--perp", type=float, default=30., help="perplexity")
    parser.add_argument("--X", type=str, default=None, help="X, data, .npy file")
    parser.add_argument("--y", type=str, default=None, help="y, labels, .npy file")

    opt = parser.parse_args()
    dens = opt.dens
    nDims = opt.nDims
    iter = opt.iter
    perp = opt.perp
    X_path = opt.X
    y_path = opt.y
    # print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # print("Running example on 2,500 MNIST digits...")
    if X_path:
        X = np.load(X_path, allow_pickle=True)[np.random.choice()]
        if y_path:
            y = np.load(y_path, allow_pickle=True)
        else:
            y = np.zeros((X.shape[0]))
    else:
        X, y = get_gaussian_data(n_clusters=4)
    
    # X = np.loadtxt("mnist2500_X.txt")
    # labels = np.loadtxt("mnist2500_labels.txt")
    
    np.random.seed(2)
    Y = tsne(X, nDims, perp, iter, dens)
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plt.scatter(X[:, 0], X[:, 1], 3, y)
    plt.subplot(1,2,2)
    plt.scatter(Y[:, 0], Y[:, 1], 3, y)
    plt.show()
