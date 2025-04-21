import numpy as np
def fama_macbeth_single_factor(exceed_ret: np.ndarray, betas: np.ndarray, resid: np.ndarray, market_ret: np.ndarray) -> tuple:
    """
    Perform Fama-MacBeth two-pass regression for the CAPM (single-factor model).

    :param exceed_ret: Matrix of excess returns (T x N)
    :param betas: Matrix of estimated betas from time-series regressions (N x 1)
    :param resid: Matrix of residuals from time-series regressions (T x N)
    :return: Risk premia (lambda), covariance matrix of alphas, J-statistic, and p-value
    """
    T, N = exceed_ret.shape
    # Step 1: Compute the average excess returns (E_t(R))
    avg_exceed_ret = np.mean(exceed_ret, axis=0).reshape(-1,1)  # 1 x N

    # Step 2: Compute the covariance matrix of residuals (Sigma)
    sigma = np.cov(resid.T)   # Covariance matrix (N x N)
    sigma_f=np.cov(market_ret).reshape(1,1)
    # Step 3: Compute lambda (CAPM risk premia) and alpha (excess return after accounting for beta)
    # (beta' * beta)^-1 * beta' * E_t(R)
    beta_matrix = betas.T @ betas
    lambda_hat = np.linalg.inv(beta_matrix) @ betas.T @ avg_exceed_ret# number
    lambda_hat_value=lambda_hat.item()
    alpha_hat = avg_exceed_ret - lambda_hat_value * betas  # N x 1

    # Step 4: Compute covariance of alpha_hat (Cov(alpha_hat))
    identity_matrix = np.eye(N)
    part1 = identity_matrix - betas @ np.linalg.inv(beta_matrix) @ betas.T
    part_1=np.linalg.inv(beta_matrix) @ betas.T
    cov_alpha = (1 / T) * part1 @ sigma @ part1.T  # Covariance matrix for alphas
    cov_lambda= (1/T) * (part_1 @ sigma @ part_1.T + sigma_f )
    # Step 5: Compute the test statistic and p-value (chi-square test)
    test_lambda=lambda_hat_value/np.sqrt(cov_lambda.item())
    test_stat = alpha_hat.T @ np.linalg.inv(cov_alpha) @ alpha_hat  # Test statistic
    p_value_alpha = 1 - chi2.cdf(test_stat, df=N - 1)  # p-value using chi-square distribution
    p_value_lambda=1 - t_dist.cdf(test_lambda,df=N-1)

    return lambda_hat, cov_alpha, test_stat, p_value_alpha, p_value_lambda

