from scipy.stats import ortho_group
import numpy as np
def get_gaussian_data(dims=2, n_clusters=5):
    X = []
    y = []

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

def get_gaussian_data_jedi(dims=2, n_clusters=5):
    X = []
    y = []
    cluster_size = []
    from scipy.stats import ortho_group

    for cluster in range(n_clusters):
        means = 100 * (np.random.rand(dims) - 0.5)
        diag = np.abs(np.diag(40 * np.random.rand(dims)))
        O = ortho_group.rvs(dims)
        covs = O.T @ diag @ O
        num_samples = np.random.randint(100,300)

        cloud = np.random.multivariate_normal(means, covs, num_samples)

        X.append(cloud)
        y.append([cluster] * num_samples)
        cluster_size.append(num_samples)
    return np.concatenate(X), np.concatenate(y), cluster_size