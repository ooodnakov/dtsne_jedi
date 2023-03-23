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

def get_dtsne_data(setting=0):
    X = []
    y = []
    n_cl = {0:3,1:3,2:3,3:10,4:5}
    for cluster in range(n_cl[setting]):
        if setting == 0:
            means = {0:[10,0],1:[0,15],2:[-10,0]}[cluster]
            covs = np.eye(2) * 2 ** (cluster)
            num_samples = {0:100,1:200,2:500}[cluster]
        if setting == 1:
            means = 50 * np.random.rand(50)
            covs =  np.eye(50) * 2
            num_samples = 200 * (cluster + 1)
        if setting == 2:
            means = 50 * np.random.rand(50)
            covs = np.eye(50) * 2 ** (cluster + 1)
            num_samples = 300
        if setting == 3:
            means = 50 * np.random.rand(50)
            covs = np.eye(50) * (cluster + 1)
            num_samples = 200
        if setting == 4:
            means = 50 * np.random.rand(150)
            covs = np.eye(150) * (cluster + 1)
            num_samples = 200
            
            
        cloud = np.random.multivariate_normal(means, covs, num_samples)
        
        X.append(cloud)
        y.append([cluster] * num_samples)
    
    return np.concatenate(X), np.concatenate(y)