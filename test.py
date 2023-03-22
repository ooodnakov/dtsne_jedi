from dtsnejedi.algorithms.dtsne import dtsne
from dtsnejedi.utils.data_gen import get_gaussian_data
import argparse
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dens", type=int, default=0, help="use dtSNE")
    parser.add_argument("--nDims", type=int, default=2, help="nDims")
    parser.add_argument("--iter", type=int, default=300, help="iterations")
    parser.add_argument("--perp", type=float, default=30., help="perplexity")
    parser.add_argument("--verbose", type=int, default=1, help="verbosity")
    parser.add_argument("--X", type=str, default=None, help="X, data, .npy file")
    parser.add_argument("--y", type=str, default=None, help="y, labels, .npy file")

    opt = parser.parse_args()
    dens = opt.dens
    nDims = opt.nDims
    iter = opt.iter
    perp = opt.perp
    verbose = opt.verbose
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
        np.random.seed(6)
        X, y = get_gaussian_data(n_clusters=4)
    
    # X = np.loadtxt("mnist2500_X.txt")
    # labels = np.loadtxt("mnist2500_labels.txt")
    
    np.random.seed(2)
    Y = dtsne(X, nDims, perp, iter, dens, verbose)
    plt.figure(figsize=(15,8))
    plt.subplot(1,2,1)
    plt.scatter(X[:, 0], X[:, 1], 3, y)
    plt.subplot(1,2,2)
    plt.scatter(Y[:, 0], Y[:, 1], 3, y)
    plt.show()
