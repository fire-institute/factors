import numpy as np
from scipy.stats import t as t_dist
#这里原假设是β=0（也就是β没有作用）
def beta_t_test(resid: np.ndarray,
                betas: np.ndarray,
                factors: np.ndarray) -> tuple:
    """
    对多因子回归中每个 β 做 t 检验 (H0: β=0)。

    :param resid: 回归残差矩阵，形状 (T, N)
    :param betas: β 系数矩阵，形状 (N, K)
    :param factors: 因子收益矩阵，形状 (T, K)
    :return:
      t_stats: t 统计量矩阵，形状 (N, K)
      p_values: 双侧 p‑value 矩阵，形状 (N, K)
    """
    T, N = resid.shape
    K = factors.shape[1]

    # 1) 构造中心化后的 X
    #    F̄_j = mean_t F_{t,j},  X_{t,j} = F_{t,j} - F̄_j
    F_bar = np.nanmean(factors, axis=0)        # (K,)
    X = factors - F_bar[np.newaxis, :]         # (T, K)

    # 2) 计算 (X'X)⁻¹ 并提取对角线
    xtx = X.T @ X                              # (K, K)
    inv_xtx = np.linalg.inv(xtx)               # (K, K)
    diag_xtx = np.diag(inv_xtx)                # (K,)

    # 3) 自由度
    df = T - K - 1

    # 4) 循环每个资产 i，先算 S²_i，再批量算出 t_stats[i,:] 和 p_values[i,:]
    t_stats = np.zeros((N, K))
    p_values = np.zeros((N, K))

    for i in range(N):
        # 4.1 残差方差估计 S²_i = sum_t eps_{t,i}² / df
        S2_i = (resid[:, i] ** 2).sum() / df

        # 4.2 标准误向量：se_j = sqrt(S²_i * inv_xtx[j,j])
        se_vec = np.sqrt(S2_i * diag_xtx)     # (K,)

        # 4.3 t 统计量和双侧 p‑value
        t_vec = betas[i, :] / se_vec
        p_vec = 2 * (1 - t_dist.cdf(np.abs(t_vec), df))

        t_stats[i, :] = t_vec
        p_values[i, :] = p_vec

    return t_stats, p_values
