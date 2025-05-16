import numpy as np
import pandas as pd

def group_sz(stock_size: pd.Series) -> pd.Series:
    '''
    Group the stocks for a given date into "small" and "big" by size.

    Parameters
    ----------
    size : pd.Series
        Series of stock size for a given date.

    Returns
    -------
    pd.Series
        A Series of group labels ('small' or 'big') with the same index as input.  
    
    '''
    # For computational performance, convert the data structure into numpy array.
    index = stock_size.index
    sz = stock_size.to_numpy().ravel()
    median = np.median(sz)
    group = pd.Series(
        np.where(sz <= median, 'small', 'big'), 
        index = index,
        name = 'size_group'
    )

    return group

def group_val(stock_value: pd.Series) -> pd.Series:
    '''
    Group the data for a given date into "growth", "neutral", and "value" by value.

    Parameters
    ----------
    value : pd.Series
        Series of stock value for a given date.

    Returns
    -------
    pd.Series
        A Series of group labels ('growth', 'neutral', 'value') indexed the same as the input. 
        - 'growth': stocks in the bottom 30% 
        - 'neutral': stocks between 30% and 70%
        - 'value': stocks in the top 30% 
    '''
    # For computational performance, convert the data structure into numpy array.
    index = stock_value.index
    val = stock_value.to_numpy().ravel()
    quantile_30 = np.quantile(val, 0.3)
    quantile_70 = np.quantile(val, 0.7)
    
    conditions = [
        val <= quantile_30,
        (val > quantile_30) & (val <= quantile_70),
        val > quantile_70
    ]
    choices = ['growth', 'neutral', 'value']

    group = pd.Series(
        np.select(conditions, choices, default = None), 
        index = index,
        name = 'value_group'
    )
    
    return group

def cal_smb_hml(
        stock_return: pd.Series, 
        stock_size: pd.Series, 
        stock_value: pd.Series
) -> tuple[float, float]:
    '''
    Calculate Fama-French SMB and HML factors for a given date.

    Parameters
    ----------
    stock_return : pd.Series
        Series of stock returns indexed by stock code.
    stock_size : pd.Series
        Series of stock market capitalizations (size), indexed by stock code.
    stock_value : pd.Series
        Series of stock value indexed by stock code.

    Returns
    -------
    smb : float
        The SMB factor: average return of small portfolios minus big portfolios.
    hml : float
        The HML factor: average return of value portfolios minus growth portfolios.
    '''
    size_group = group_sz(stock_size)
    value_group = group_val(stock_value)

    df = pd.concat([stock_return, size_group, value_group], axis = 1)
    df.columns = ['stock_return', 'size_group', 'value_group']

    groups = {}
    for sz in ['small', 'big']:
        for val in ['growth', 'neutral', 'value']:
            condition = (df['size_group'] == sz) & (df['value_group'] == val)
            groups[f'{sz}_{val}'] = np.nanmean(df.loc[condition, 'stock_return'])

    smb = (groups['small_growth'] + groups['small_neutral'] + groups['small_value']) / 3 \
        - (groups['big_growth'] + groups['big_neutral'] + groups['big_value']) / 3

    hml = (groups['small_value'] + groups['big_value']) / 2 \
        - (groups['small_growth'] + groups['big_growth']) / 2

    return smb, hml

def cal_mkt_ret(
        stock_return: pd.Series, 
        stock_size: pd.Series, 
        risk_free_rate: float
) -> float:
    '''
    Calculate the excess market return using value-weighted returns.

    Parameters
    ----------
    stock_return : pd.Series
        Series of stock returns for a given date, indexed by stock code.
    stock_size : pd.Series
        Series of market capitalizations for the same stocks, indexed by stock code.
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

def cal_ff3(
        stock_return: pd.Series, 
        stock_size: pd.Series, 
        stock_value: pd.Series, 
        risk_free_rate: float
) -> tuple[float, float, float]:
    '''
    Calculate Fama-French Three Factors (SMB, HML, MKT-RF) for a given date.

    Parameters
    ----------
    stock_return : pd.Series
        Series of stock returns for a given date, indexed by stock code.
    stock_size : pd.Series
        Series of stock market capitalizations (size), indexed by stock code.
    stock_value : pd.Series
        Series of stock value indexed by stock code.
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