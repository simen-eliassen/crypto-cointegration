import pandas as pd
import datetime
import statsmodels.api as sm
from arch.unitroot import *
import numpy as np
import tqdm
import itertools
import requests
import json
import multiprocessing
import warnings
import sys

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def adf_pp_kpss(df: pd.DataFrame):
    """This code will perform both the Augmented Dickey-Fuller test
    and the Kwiatkowski-Phillips-Schmidt-Shin test on the price series,
    and return the test statistic and p-value for each test. The null
    hypothesis for both tests is that there is a unit root present in
    the observable price series, and the alternative hypothesis is that
    there is no unit root present. The p-value can be used to determine
    whether to reject or fail to reject the null hypothesis. A common
    threshold for rejecting the null hypothesis is a p-value of 0.05 or
    less.

    It's important to notice that the ADF and KPSS are different tests and they 
    are not meant to be used together, they test for different kind of stationarity. 
    The ADF test is based on the assumption that the series is non-stationary, 
    while the KPSS test is based on the assumption that the series is stationary.

    Parameters
    ----------
    df : pd.DataFrame
        Input price data with crypto symbols as columns

    """
    stationary_tests = [ADF, PhillipsPerron, KPSS]
    dfs = []

    for col in [c for c in df.columns if c != "date"]:
        for test in stationary_tests:
            input_data = df[col].dropna()
            t = test(input_data)

            data = {
                "Symbol": col,
                "TestName": t._test_name,
                "PValue": t.pvalue,
                "NullHypothesis": t.null_hypothesis,
                "AlternativeHypothesis": t.alternative_hypothesis,
            }

            df_tmp = pd.DataFrame(data, index=[0])

            dfs.append(df_tmp)

    df_out = pd.concat(dfs)
    return df_out


def perform_stationary_test_residuals(data_list):
    df_pivot = data_list[1]
    symbols=list(data_list[0])
    from_date = datetime.datetime(2022, 1, 1)
    df = df_pivot[["date"] + symbols].dropna()
    df = df[df.date >= from_date]
    try:
        X = sm.add_constant(df[symbols[:-1]])
        y = df[symbols[-1]]
        model = sm.OLS(y, X).fit()
        df_tmp = pd.DataFrame(model.resid,columns=['_'.join(symbols)])
        df_test = perform_stationary_tests(df_tmp)
    except Exception:
        df_test = pd.DataFrame()
        
    return df_test


# Read historical data and filter on top coins
df = pd.read_csv("./data/all_binance_1d.csv")
df["symbol_short"] = df.symbol.str.rstrip("USDT")
df_keep = df[df.symbol_short.isin(top100)].copy()

# Create pivot table
df_pivot = df_keep.pivot_table(
    values="close_price", index=["date"], columns="symbol"
).reset_index()

for c in df_pivot.columns:
    if c != 'date':
        df_pivot[f"{c}_diff"] = df_pivot[c].copy().diff()

df_pivot.date = pd.to_datetime(df_pivot.date)

# profile = ProfileReport(df_pivot[[c for c in df_pivot.columns if 'diff' in c]], tsmode=True, sortby="date", config_file="custom.yml")
# profile.to_file("report.html")

# Perform regular stationary test
df_tests = perform_stationary_tests(df_pivot)

# Filter on 6 tests passed
conditions = [
    (
        (df_tests.PValue > 0.1)
        & (~df_tests.Symbol.str.contains("_diff"))
        & (df_tests.TestName != "KPSS Stationarity Test")
    ),
    (
        (df_tests.PValue < 0.05)
        & (~df_tests.Symbol.str.contains("_diff"))
        & (df_tests.TestName == "KPSS Stationarity Test")
    ),
    (
        (df_tests.PValue < 0.05)
        & (df_tests.Symbol.str.contains("_diff"))
        & (df_tests.TestName != "KPSS Stationarity Test")
    ),
    (
        (df_tests.PValue > 0.1)
        & (df_tests.Symbol.str.contains("_diff"))
        & (df_tests.TestName == "KPSS Stationarity Test")
    ),
]

choices = [1,1,1,1]
df_tests['Keep'] = np.select(conditions, choices, default=0)


if __name__ == '__main__':
    # Create all combinations
    df_tests.Symbol = df_tests.Symbol.str.replace('_diff','')
    df_tests = df_tests.groupby(['Symbol'])['Keep'].sum().reset_index()
    all_symbols = df_tests[df_tests.Keep==6].Symbol.to_list()
    output_list = []
    for dep_coin in tqdm.tqdm(all_symbols):
        for i in range(1,4):
            if dep_coin in ['BTCUSDT']:
                symbols2combine = [s for s in all_symbols if s!=dep_coin]
                combs = list(itertools.combinations(symbols2combine, i))
                output_list = output_list + [c+(dep_coin,) for c in combs]
            
    output_list = [[c, df_pivot] for c in output_list]
    
    with multiprocessing.Pool(processes=6) as pool:
        #results = pool.map(perform_stationary_test_residuals, output_list)
        results = list(tqdm.tqdm(pool.imap(perform_stationary_test_residuals, output_list), total=len(output_list)))
        
    df = pd.concat(results)
    df.to_excel('./stationary_test_residuals.xlsx')
    sys.exit()


