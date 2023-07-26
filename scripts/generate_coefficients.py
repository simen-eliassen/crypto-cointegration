# Import libraries
import pandas as pd
import logging
import datetime
import tqdm
import os
import sqlite3
import numpy as np
import itertools
import pdcast as pdc
import statsmodels.api as sm
from dotenv import load_dotenv

# Load environment variables
_ = load_dotenv()
working_dir = os.getenv("WORKING_DIR")

# Change to working dir
os.chdir(working_dir)

# Import custom modules
from utils.custom_logger import initiate_logger
from utils.database_tools import query_stationary_test, query_topN_price_data
from utils.stationary_tests import filter_adf_pp_kpss, adf_pp_kpss
from utils.calculate_residuals import calc_resid
from utils import create_models


df_stationary_tests = query_stationary_test()
fk_symbols = filter_adf_pp_kpss(df_stationary_tests)
df_price_data = query_topN_price_data(100)
df_price_data = pdc.downcast(df_price_data)

with sqlite3.connect("./data/database.db") as conn:
    df_symbols = pd.read_sql_query(
        "select pk_symbols as fk_symbols, id from DimSymbols", conn
    )

df_symbols = df_symbols.merge(
    pd.DataFrame(fk_symbols, columns=["fk_symbols"]),
    on="fk_symbols",
    how="inner",
)

all_symbols = df_symbols.id.unique()

crypto_pairs = create_models.generate_crypto_pairs(
    dependent_coin="BTCUSDT",
    all_symbols=all_symbols,
    min_pairs=2,
    max_pairs=4,
)

# Create pivot table
df_pivot = df_price_data.pivot_table(
    values="close_price", index=["date"], columns="id"
).reset_index()

# Create list to iterate over if more than 200 data points
iter_list = []
for crypto_pair in tqdm.tqdm(crypto_pairs):
    cols2keep = ["date"] + list(crypto_pair)
    df_tmp = df_pivot[cols2keep].dropna()
    if len(df_tmp) > 200:
        iter_list.append(df_tmp)


residuals = []
for df in iter_list:
    df_tmp = calc_resid(df)
    residuals.append(df_tmp)


# Concatenate all residuals
df_all = pd.concat(residuals)

# Run stationary tests on residuals
df_stationary_tests_resid = adf_pp_kpss(df_all)

# Filter on at least 5 out of 6 tests passed
conditions = [
    (
        (df_stationary_tests_resid.PValue < 0.05)
        & (df_stationary_tests_resid.TestName == "Augmented Dickey-Fuller")
    ),
    (
        (df_stationary_tests_resid.PValue < 0.05)
        & (df_stationary_tests_resid.TestName == "Phillips-Perron Test")
    ),
    (
        (df_stationary_tests_resid.PValue > 0.1)
        & (df_stationary_tests_resid.TestName == "KPSS Stationarity Test")
    ),
]

choices = [1, 1, 1]
df_stationary_tests_resid["Keep"] = np.select(conditions, choices, default=0)
df_final = (
    df_stationary_tests_resid.groupby(["TimeSeriesName"])["Keep"].sum().reset_index()
)

df_final = df_final[df_final.Keep == 3]
models_to_make = df_final.TimeSeriesName.apply(
    lambda row: row.split("_")[:-1]
).to_list()


# Create pivot table
df_pivot = df_price_data.pivot_table(
    values="close_price",
    index=["date"],
    columns="id",
).reset_index()

models = []
for symbol_list in tqdm.tqdm(models_to_make):
    model_output = create_models.linear_model(
        df=df_pivot[symbol_list], symbol_list=symbol_list
    )
    models.append(model_output)

df_models = pd.concat(models)
df_models = df_models.reset_index(drop=True)
df_models.to_parquet("./data/models.parquet")


