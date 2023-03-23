
import numpy as np
from tqdm import tqdm
from ..utils.utils import Hbeta, pca



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
    sumP = max(sum(P),np.finfo(np.double).eps)
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
    for i in tqdm(range(n)):

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
        gamma /= gamma.max()
    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P, gamma


def dtsne(X=np.array([]), n_components=2, perplexity=30.0, n_iter=1000, dens=False, verbose=1, random_seed=None, initial_dims=None):
    
    if random_seed:
        np.random.seed(random_seed)
    # Check inputs
    if not isinstance(n_components, int):
        print("Error: array X should have type float.")
        return -1
    if dens:
        print("Runs dtSNE variant of algorithm.")
    # Initialize variables
    if initial_dims:
        X = pca(X, initial_dims).real
    (n, d) = X.shape
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = pca(X, n_components).real
    Y = 1e-4 * Y / np.std(Y)
    dY = np.zeros((n, n_components))
    iY = np.zeros((n, n_components))
    gains = np.ones((n, n_components))

    # Compute P-values
    P, gamma = x2p(X, 1e-5, perplexity, dens)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    # Early exaggeration
    P = P * 4.  
    P = np.maximum(P, np.finfo(np.double).eps)
    
    C_old = np.sum(P)+100
    # Run iterations
    print('Performing optimization...')
    for iter in tqdm(range(n_iter)):

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
                dY[i, :] = 4 * np.sum(np.tile(PQ[:, i] * num[:, i] * gamma[:, i], (n_components, 1)).T * (Y[i, :] - Y) , 0)
        else:
            for i in range(n):
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (n_components, 1)).T * (Y[i, :] - Y), 0)

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
        
        if (iter + 1) % 50 == 0 and verbose > 1:
            C = np.sum(P * np.log(P / Q))
            if abs(C-C_old)<5e-5 and iter > 100:
                print("Early stopping due to small change", C, C_old)
                break
            C_old = C
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.
    
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
    
