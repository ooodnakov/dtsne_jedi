from dtsnejedi.algorithms.dtsne import dtsne
from dtsnejedi.algorithms.jedi import jedi
from dtsnejedi.utils.data_gen import get_gaussian_data,get_gaussian_data_jedi
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default='tsne', help="which algorithm to use")
    parser.add_argument("--n_comp", type=int, default=2, help="n_components")
    parser.add_argument("--n_iter", type=int, default=300, help="iterations")
    parser.add_argument("--perp", type=float, default=30., help="perplexity")
    parser.add_argument("--alpha", type=float, default=0.5, help="alpha for jedi")
    parser.add_argument("--beta", type=float, default=0.5, help="beta for jedi")
    parser.add_argument("--verb", type=int, default=1, help="verbosity")
    parser.add_argument("--X", type=str, default=None, help="X, data, .npy file")
    parser.add_argument("--Z", type=str, default=None, help="Z, information matrix, .npy file")
    parser.add_argument("--y", type=str, default=None, help="y, labels, .npy file")

    opt = parser.parse_args()
    algorithm = opt.algo
    n_components = opt.n_comp
    n_iter = opt.n_iter
    perplexity = opt.perp
    beta = opt.beta
    alpha = opt.alpha
    verbose = opt.verb
    X_path = opt.X
    Z_path = opt.Z
    y_path = opt.y
    # print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    # print("Running example on 2,500 MNIST digits...")
    if X_path:
        X = np.load(X_path, allow_pickle=True)
        if algorithm=='jedi':
            if Z_path:
                Z = np.load(Z_path, allow_pickle=True)
            else:
                sys.exit('Z information matrix is required in case of JEDI algorithm')
        if y_path:
            y = np.load(y_path, allow_pickle=True)
        else:
            y = np.zeros((X.shape[0]))
    else:
        np.random.seed(6)
        if algorithm=='jedi':
            X, y, n_samples = get_gaussian_data_jedi(n_clusters=4)
            sum_X = np.sum(np.square(X), 1)
            D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
            Z = np.random.randn(X.shape[0], X.shape[0]) ** 2 
            Z = Z + Z.T
            Z[0:n_samples[0] + n_samples[1], 0:n_samples[0] + n_samples[1]] = D[0:n_samples[0] + n_samples[1], 0:n_samples[0] + n_samples[1]]
        else:    
            X, y = get_gaussian_data(n_clusters=4)
    
    # X = np.loadtxt("mnist2500_X.txt")
    # labels = np.loadtxt("mnist2500_labels.txt")
    
    np.random.seed(2)
    if algorithm=='tsne':
        Y = dtsne(X, n_components, perplexity, n_iter, 0, verbose)
    if algorithm=='dtsne':
        Y = dtsne(X, n_components, perplexity, n_iter, 1, verbose)
    if algorithm=='jedi':
        Y = jedi(X, Z, alpha, beta, n_components, perplexity, n_iter, 0, verbose)
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    sc = plt.scatter(X[:, 0], X[:, 1], 3, y)
    plt.legend(*sc.legend_elements(), title="Clusters")
    plt.subplot(1,2,2)
    sc = plt.scatter(Y[:, 0], Y[:, 1], 3, y)
    plt.legend(*sc.legend_elements(), title="Clusters")
    plt.show()
