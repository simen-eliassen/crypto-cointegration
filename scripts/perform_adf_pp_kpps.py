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


class StationaryTestPriceData:
    def __init__(self, database, topN=100):
        self.database = database
        self.topN = topN

    def run_stationary_test(self):
        # Query price data
        df_price_data = query_topN_price_data(self.topN)

        # Run stationary test
        df_stationary_test = adf_pp_kpss(df_price_data)

        # Add symbol column
        df_stationary_test["symbol"] = df_stationary_test.TimeSeriesName.str.split(
            "_", expand=True
        ).iloc[:, 0]

        # Add from_date and to_date columns
        df_min_max = (
            df_price_data[["symbol", "fk_symbols", "date"]]
            .groupby(["symbol", "fk_symbols"])
            .agg(["min", "max"])
            .reset_index()
        )
        df_min_max.columns = ["symbol", "fk_symbols", "from_date", "to_date"]
        df_stationary_test = df_stationary_test.merge(
            df_min_max, on="symbol", how="left"
        )

        # Drop symbol column
        df_stationary_test = df_stationary_test.drop(columns=["symbol"])

        # Add test_date column
        df_stationary_test["test_date"] = datetime.datetime.now()

        # Write results to database
        with sqlite3.connect(self.database) as conn:
            df_stationary_test.to_sql(
                "tblStationaryTest", conn, if_exists="replace", index=False
            )


if __name__ == "__main__":
    database = "./data/database.db"
    topN = 100

    stationary_test = StationaryTestPriceData(database, topN)
    stationary_test.run_stationary_test()
