import metrics
import numpy as np
from scipy.stats import ortho_group

data = np.random.rand(1500, 3)
O = ortho_group.rvs(3)
trasformed_data = data @ O
rho, rho_knn, rho_r = metrics.reconstruction_quality(data, trasformed_data)

assert np.isclose(rho, 1)
assert np.isclose(rho_knn, 1)
assert np.isclose(rho_r, 1)