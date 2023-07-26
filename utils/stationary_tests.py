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

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


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

    # Create pivot table
    df = df.pivot_table(
        values="close_price", index=["date"], columns="symbol"
    ).reset_index()

    for c in df.columns:
        if c != "date":
            df[f"{c}_diff"] = df[c].copy().diff()

    df.date = pd.to_datetime(df.date)

    stationary_tests = [ADF, PhillipsPerron, KPSS]
    dfs = []

    for col in tqdm.tqdm([c for c in df.columns if c != "date"]):
        for test in stationary_tests:
            input_data = df[col].dropna()
            t = test(input_data)

            data = {
                "TimeSeriesName": col,
                "TestName": t._test_name,
                "PValue": t.pvalue,
                "NullHypothesis": t.null_hypothesis,
                "AlternativeHypothesis": t.alternative_hypothesis,
            }

            df_tmp = pd.DataFrame(data, index=[0])

            dfs.append(df_tmp)

    df_out = pd.concat(dfs)
    return df_out


def filter_adf_pp_kpss(df_stationary_test):
    # Filter on 6 tests passed
    conditions = [
        (
            (df_stationary_test.PValue > 0.1)
            & (~df_stationary_test.TimeSeriesName.str.contains("_diff"))
            & (df_stationary_test.TestName != "KPSS Stationarity Test")
        ),
        (
            (df_stationary_test.PValue < 0.05)
            & (~df_stationary_test.TimeSeriesName.str.contains("_diff"))
            & (df_stationary_test.TestName == "KPSS Stationarity Test")
        ),
        (
            (df_stationary_test.PValue < 0.05)
            & (df_stationary_test.TimeSeriesName.str.contains("_diff"))
            & (df_stationary_test.TestName != "KPSS Stationarity Test")
        ),
        (
            (df_stationary_test.PValue > 0.1)
            & (df_stationary_test.TimeSeriesName.str.contains("_diff"))
            & (df_stationary_test.TestName == "KPSS Stationarity Test")
        ),
    ]

    choices = [1, 1, 1, 1]
    df_stationary_test["Keep"] = np.select(conditions, choices, default=0)
    df_stationary_test = (
        df_stationary_test.groupby(["fk_symbols"])["Keep"].sum().reset_index()
    )
    fk_symbols = df_stationary_test[df_stationary_test.Keep == 6].fk_symbols.to_list()
    return fk_symbols


def residuals_ols(df):
    """
    This function will perform the stationary tests on the residuals of the OLS model
    """
    symbols = [col for col in df.columns if col != "date"]

    try:
        X = sm.add_constant(df[symbols[:-1]])
        y = df[symbols[-1]]
        model = sm.OLS(y, X).fit()
        df_tmp = pd.DataFrame(model.resid, columns=["_".join(symbols)])
    except Exception:
        df_test = pd.DataFrame()

    return df_test
