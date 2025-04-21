import numpy as np
from scipy.stats import f
def grs_test(resid: np.ndarray, alpha: np.ndarray, factors: np.ndarray) -> tuple:
    """ Perform the Gibbons, Ross and Shanken (1989) test.
    :param resid: Matrix of residuals from the OLS of size TxN.
    :param alpha: Vector of alphas from the OLS of size Nx1.
    :param factors: Matrix of factor returns of size TxK.
    :return: Test statistic and p-value of the test.
    """
    T, N = resid.shape
    K = factors.shape[1]

    # Residual covariance estimate
    mCov = resid.T.dot(resid) / (T - K - 1)
    vMuRF = np.nanmean(factors, axis=0).reshape(1, K)
    mMuRF = np.repeat(vMuRF, T, axis=0)
    mCovRF = (factors - mMuRF).T @ (factors - mMuRF) / (T - 1)

    num = (T / N) * ((T - N - K) / (T - K - 1))
    quad = alpha.T @ np.linalg.inv(mCov) @ alpha
    denom = 1 + (vMuRF @ np.linalg.inv(mCovRF) @ vMuRF.T)
    grs_stat = float(num * quad / denom)
    p_val = 1 - f.cdf(grs_stat, N, T - N - 1)
    return grs_stat, p_val