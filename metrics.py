import numpy as np
from scipy import stats as sps
import matplotlib.pyplot as plt


def rho(pts1, pts2):
    assert len(pts1.shape) == 2, "pts1 must be 2-d array"
    assert len(pts2.shape) == 2, "pts2 must be 2-d array"
    assert pts1.shape[0] == pts2.shape[0], "arrays pts1, pts2 must have the same number of points"
    n_points = pts1.shape[0]
    dim1 = pts1.shape[1]
    dim2 = pts2.shape[1]
    dists1 = np.zeros((n_points, n_points, dim1))
    dists2 = np.zeros((n_points, n_points, dim2))

    dists1 += pts1[:, None, :]
    dists1 -= pts1[None, :, :]
    dists1 = np.linalg.norm(dists1, axis=-1)

    dists2 += pts2[:, None, :]
    dists2 -= pts2[None, :, :]
    dists2 = np.linalg.norm(dists2, axis=-1)

    stat, _ = sps.spearmanr(dists1, dists2, axis=None)
    return stat


def rho_r(pts1, pts2, k=100):
    assert len(pts1.shape) == 2, "pts1 must be 2-d array"
    assert len(pts2.shape) == 2, "pts2 must be 2-d array"
    assert pts1.shape[0] == pts2.shape[0], "arrays pts1, pts2 must have the same number of points"
    n_points = pts1.shape[0]
    k = min(k, n_points-1)
    dim1 = pts1.shape[1]
    dim2 = pts2.shape[1]
    dists1 = np.zeros((n_points, n_points, dim1))
    dists2 = np.zeros((n_points, n_points, dim2))

    dists1 += pts1[:, None, :]
    dists1 -= pts1[None, :, :]
    dists1 = np.linalg.norm(dists1, axis=-1)

    dists2 += pts2[:, None, :]
    dists2 -= pts2[None, :, :]
    dists2 = np.linalg.norm(dists2, axis=-1)

    sorted_dists1 = np.sort(dists1, axis=0)
    sorted_dists2 = np.sort(dists2, axis=0)
    radii1 = sorted_dists1[k]
    radii2 = sorted_dists2[k]

    stat, _ = sps.spearmanr(radii1, radii2, axis=None)
    return stat


def rho_knn(pts1, pts2, k=100):
    assert len(pts1.shape) == 2, "pts1 must be 2-d array"
    assert len(pts2.shape) == 2, "pts2 must be 2-d array"
    assert pts1.shape[0] == pts2.shape[0], "arrays pts1, pts2 must have the same number of points"
    n_points = pts1.shape[0]
    k = min(k, n_points-1)
    dim1 = pts1.shape[1]
    dim2 = pts2.shape[1]
    dists1 = np.zeros((n_points, n_points, dim1))
    dists2 = np.zeros((n_points, n_points, dim2))

    dists1 += pts1[:, None, :]
    dists1 -= pts1[None, :, :]
    dists1 = np.linalg.norm(dists1, axis=-1)

    dists2 += pts2[:, None, :]
    dists2 -= pts2[None, :, :]
    dists2 = np.linalg.norm(dists2, axis=-1)

    sorted_dists1 = np.sort(dists1, axis=0)
    sorted_dists2 = np.sort(dists2, axis=0)
    k_nearest_neigbours1 = sorted_dists1[1:k+1]
    k_nearest_neigbours2 = sorted_dists2[1:k+1]

    stat, _ = sps.spearmanr(k_nearest_neigbours1,
                            k_nearest_neigbours2,
                            axis=None)
    return stat


def reconstruction_quality(pts1, pts2, k=100, plot=False):
    assert len(pts1.shape) == 2, "pts1 must be 2-d array"
    assert len(pts2.shape) == 2, "pts2 must be 2-d array"
    assert pts1.shape[0] == pts2.shape[0], "arrays pts1, pts2 must have the same number of points"
    n_points = pts1.shape[0]
    k = min(k, n_points-1)
    dim1 = pts1.shape[1]
    dim2 = pts2.shape[1]
    dists1 = np.zeros((n_points, n_points, dim1))
    dists2 = np.zeros((n_points, n_points, dim2))

    dists1 += pts1[:, None, :]
    dists1 -= pts1[None, :, :]
    dists1 = np.linalg.norm(dists1, axis=-1)

    dists2 += pts2[:, None, :]
    dists2 -= pts2[None, :, :]
    dists2 = np.linalg.norm(dists2, axis=-1)

    sorted_dists1 = np.sort(dists1, axis=0)
    sorted_dists2 = np.sort(dists2, axis=0)
    k_nearest_neigbours1 = sorted_dists1[1:k+1]
    k_nearest_neigbours2 = sorted_dists2[1:k+1]
    radii1 = sorted_dists1[k]
    radii2 = sorted_dists2[k]

    rho, _ = sps.spearmanr(dists1, dists2, axis=None)
    rho_knn, _ = sps.spearmanr(k_nearest_neigbours1,
                               k_nearest_neigbours2,
                               axis=None)
    rho_r, _ = sps.spearmanr(radii1, radii2, axis=None)
    if plot:
        plt.title("Reconstruction quality", fontsize=15)
        plt.bar([0, 1, 2], [rho, -rho_knn, rho_r], width=0.5,
                color=["r", "g", "b"],
                label=[r'$\rho$', r'$\rho_{knn}$', r'$\rho_{r}$'])
        plt.xlim(-0.75, 2.75)
        plt.ylim(-1, 1)
        plt.ylabel("Correlation")
        plt.xticks([0, 1, 2], ["global\nreconstruction",
                               "local\nreconstruction",
                               "relative\ndensity\nreconstruction"])
        plt.legend()
        plt.show()
    return rho, rho_knn, rho_r


def neighbourhood_overlap_score(pts1, pts2, plot=False):
    assert len(pts1.shape) == 2, "pts1 must be 2-d array"
    assert len(pts2.shape) == 2, "pts2 must be 2-d array"
    assert pts1.shape[0] == pts2.shape[0], "arrays pts1, pts2 must have the same number of points"
    
    def helper(l1, l2):
        number_of_common_neighbours = np.zeros(n_points)
        set1 = set()
        set2 = set()
        for k in range(1, n_points):
            number_of_common_neighbours[k] = number_of_common_neighbours[k-1]
            if k > 2:
                number_of_common_neighbours[k-1] /= (k-1)
            if l1[k] in set2:
                number_of_common_neighbours[k] += 1
            set1.add(l1[k])
            if l2[k] in set1:
                number_of_common_neighbours[k] += 1
            set2.add(l2[k])
        number_of_common_neighbours[-1] /= (n_points-1)
        return number_of_common_neighbours

    n_points = pts1.shape[0]
    dim1 = pts1.shape[1]
    dim2 = pts2.shape[1]
    dists1 = np.zeros((n_points, n_points, dim1))
    dists2 = np.zeros((n_points, n_points, dim2))

    dists1 += pts1[:, None, :]
    dists1 -= pts1[None, :, :]
    dists1 = np.linalg.norm(dists1, axis=-1)

    dists2 += pts2[:, None, :]
    dists2 -= pts2[None, :, :]
    dists2 = np.linalg.norm(dists2, axis=-1)
    
    sorted_dists1 = np.argsort(dists1, axis=0)
    sorted_dists2 = np.argsort(dists2, axis=0)
    
    nos = np.zeros((n_points, n_points))
    for i in range(n_points):
        nos[i] = helper(sorted_dists1[:, i], sorted_dists2[:, i])
    nos = nos.mean(axis=0)
    if plot:
        plt.title("Neighbourhood overlap score", fontsize=15)
        xs = np.linspace(0., 1., n_points)[1:]
        plt.plot(xs, nos[1:], label="NOS")
        plt.hlines(1, 0, 1, linestyle='--', color='black', alpha=0.5)
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        plt.xlabel("Neighbourhood size in % of data")
        plt.ylabel("Neighbourhood overlap in %")
        plt.legend(loc="lower right")
        plt.show()
    return nos


if __name__=='__main__':
    data = np.random.rand(1500, 3)
    O = sps.ortho_group.rvs(3)
    transformed_data = data @ O
    rho, rho_knn, rho_r = reconstruction_quality(data, transformed_data)
    print('Calculated metrics:', rho, rho_knn, rho_r)
    assert np.isclose(rho, 1)
    assert np.isclose(rho_knn, 1)
    assert np.isclose(rho_r, 1)