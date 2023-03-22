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
