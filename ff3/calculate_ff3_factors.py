import numpy as np
from typing import Tuple
from numba import njit

@njit
def group_sz(stock_size: np.ndarray) -> np.ndarray:
    '''
    Group the stocks for a given date into "small" and "big" by size. 
    
    Parameters
    ----------
    stock_size : np.ndarray
        Array of stock size for a given date.

    Returns
    -------
    np.ndarray
        Integer array of group labels where:
        - 0 represents 'small' (size ≤ median)
        - 1 represents 'big' (size > median)
    '''
    median = np.median(stock_size)
    group = np.where(stock_size <= median, 0, 1).astype(np.float64)

    return group

@njit
def group_val(stock_value: np.ndarray) -> np.ndarray:
    '''
    Group the data for a given date into "growth", "neutral", and "value" by value.

    Parameters
    ----------
    stock_value : np.ndarray
        Array of stock b/m for a given date.

    Returns
    -------
    np.ndarray
        Integer array of group labels where:
        - 0 represents 'growth' (value ≤ 30th percentile)
        - 1 represents 'neutral' (30th < value ≤ 70th percentile) 
        - 2 represents 'value' (value > 70th percentile)
    '''
    quantile_30 = np.quantile(stock_value, 0.3)
    quantile_70 = np.quantile(stock_value, 0.7)

    n = stock_value.shape[0]
    group = np.empty(n, dtype = np.float64)
    for i in range(n):
        if stock_value[i] <= quantile_30:
            group[i] = 0
        elif stock_value[i] <= quantile_70:
            group[i] = 1
        else:
            group[i] = 2

    return group

@njit
def cal_smb_hml(
    stock_return: np.ndarray,
    stock_size: np.ndarray,
    stock_value: np.ndarray
) -> tuple[float, float]:
    '''
    Calculate Fama-French SMB and HML factors for a given date.

    Parameters
    ----------
    stock_return : np.ndarray
        Array of stock returns.
    stock_size : np.ndarray
        Array of stock market capitalizations (size).
    stock_value : np.ndarray
        Array of stock b/m.

    Returns
    -------
    smb : float
        The SMB factor: average return of small portfolios minus big portfolios.
    hml : float
        The HML factor: average return of value portfolios minus growth portfolios.
    '''
    n = stock_return.shape[0]
    df = np.empty((n, 3), dtype = np.float64)
    df[:, 0] = stock_return
    df[:, 1] = group_sz(stock_size)
    df[:, 2] = group_val(stock_value)

    # [size, value] portfolio mean 
    group_returns = np.full((2, 3), np.nan)
    for sz in [0, 1]:
        for val in [0, 1, 2]:   
            condition = (df[:, 1] == sz) & (df[:, 2] == val)
            group_returns[sz, val] = np.nanmean(df[condition, 0])

    smb = np.nanmean(group_returns[0]) - np.nanmean(group_returns[1])
    hml = np.nanmean(group_returns[:, 2]) - np.nanmean(group_returns[:, 0])

    return smb, hml

@njit
def cal_mkt_ret(
        stock_return: np.ndarray, 
        stock_size: np.ndarray, 
        risk_free_rate: float
) -> float:
    '''
    Calculate the excess market return using value-weighted returns.

    Parameters
    ----------
    stock_return : np.ndarray
        Array of stock returns for a given date.
    stock_size : np.ndarray
        Array of market capitalizations for the same stocks.
    risk_free_rate : float
        The risk-free rate for the given date.

    Returns
    -------
    mkt_ret : float
        The value-weighted market excess return.
    '''
    weights = stock_size / stock_size.sum()
    weighted_return = np.sum(weights * stock_return)
    mkt_ret = weighted_return - risk_free_rate

    return mkt_ret

@njit
def cal_ff3(
        stock_return: np.ndarray, 
        stock_size: np.ndarray, 
        stock_value: np.ndarray, 
        risk_free_rate: float
) -> tuple[float, float, float]:
    '''
    Calculate Fama-French Three Factors (SMB, HML, MKT-RF) for a given date.

    Parameters
    ----------
    stock_return : np.ndarray
        Array of stock returns for a given date, indexed by stock code.
    stock_size : np.ndarray
        Array of stock market capitalizations (size), indexed by stock code.
    stock_value : np.ndarray
        Array of stock value indexed by stock code.
    risk_free_rate : float
        Risk-free rate for the given date.

    Returns
    -------
    smb : float
        The SMB factor: average return of small portfolios minus big portfolios.
    hml : float
        The HML factor: average return of value portfolios minus growth portfolios.
    mkt_ret : float
        The value-weighted market excess return.
    '''
    smb, hml = cal_smb_hml(stock_return, stock_size, stock_value)
    mkt_ret = cal_mkt_ret(stock_return, stock_size, risk_free_rate)

    return smb, hml, mkt_ret

@njit
def batch_cal_ff3(
    stock_return: np.ndarray, 
    stock_size: np.ndarray, 
    stock_value: np.ndarray, 
    risk_free_rate: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Batch calculate Fama-French 3 factors (SMB, HML, Mkt-Rf) for multiple dates.
    
    Parameters
    ----------
    stock_return : np.ndarray
        2D array of stock returns with shape (trade_date, stock_code).
        Each row represents a trading date.
    stock_size : np.ndarray
        2D array of market capitalizations with shape (trade_date, stock_code).
        Must align with stock_return.
    stock_value : np.ndarray
        2D array of value factors with shape (trade_date, stock_code).
        Must align with stock_return.
    risk_free_rate : np.ndarray
        1D array of risk-free rates with shape (n_dates,).
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing three 1D arrays:
        - smb: Small Minus Big factor series (n_dates,)
        - hml: High Minus Low factor series (n_dates,)
        - mkt_ret: Market excess return series (n_dates,)
    '''
    num_of_dates = stock_return.shape[0]
    smb = np.empty(num_of_dates, dtype = np.float64)
    hml = np.empty(num_of_dates, dtype = np.float64)
    mkt_ret = np.empty(num_of_dates, dtype = np.float64)
    for date in range(num_of_dates):
        stock_return_date = stock_return[date]
        stock_size_date = stock_size[date]
        stock_value_date = stock_value[date]
        risk_free_rate_date = risk_free_rate[date].item()

        smb_date, hml_date, mkt_ret_date = cal_ff3(
            stock_return_date, 
            stock_size_date,
            stock_value_date,
            risk_free_rate_date
        )

        smb[date] = smb_date
        hml[date] = hml_date
        mkt_ret[date] = mkt_ret_date

    return smb, hml, mkt_ret


