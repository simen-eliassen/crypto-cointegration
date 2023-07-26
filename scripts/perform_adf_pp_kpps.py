# Import libraries
import pandas as pd
import logging
import datetime
import tqdm
import os
import sqlite3
import numpy as np
from dotenv import load_dotenv

# Load environment variables
_ = load_dotenv()
working_dir = os.getenv("WORKING_DIR")

# Change to working dir
os.chdir(working_dir)

# Import custom modules
from utils.custom_logger import initiate_logger
from utils.database_tools import query_topN_price_data
from utils.stationary_tests import adf_pp_kpss

df_price_data = query_topN_price_data(100)
df_stationary_test = adf_pp_kpss(df_price_data)
df_stationary_test["symbol"] = df_stationary_test.TimeSeriesName.str.split("_", expand=True).iloc[:, 0]
df_min_max = df_price_data[['symbol','fk_symbols','date']].groupby(['symbol','fk_symbols']).agg(['min','max']).reset_index()
df_min_max.columns = ['symbol','fk_symbols','from_date','to_date']
df_stationary_test = df_stationary_test.merge(df_min_max, on='symbol', how='left')
df_stationary_test = df_stationary_test.drop(columns=["symbol"])
df_stationary_test["test_date"] = datetime.datetime.now()

with sqlite3.connect("./data/database.db") as conn:
    df_stationary_test.to_sql("tblStationaryTest", conn, if_exists="append", index=False)

