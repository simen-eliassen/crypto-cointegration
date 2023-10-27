import datetime
import itertools
import logging
import os
import sqlite3
import warnings


import numpy as np
import pandas as pd
import pdcast as pdc
import statsmodels.api as sm
import tqdm
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Suppress all warning messages
warnings.filterwarnings("ignore")

# Load environment variables
_ = load_dotenv()
working_dir = os.getenv("WORKING_DIR")

# Change to working dir
os.chdir(working_dir)

# Import custom modules
from utils import create_models
from utils.calculate_residuals import calc_resid
from utils.custom_logger import initiate_logger
from utils.database_tools import (
    query_stationary_test,
    query_topN_price_data,
    query_topN_symbols,
    query_topN_symbols_with_last_date,
)
from utils.stationary_tests import adf_pp_kpss, filter_adf_pp_kpss


class CryptocurrencyModelGenerator:
    def __init__(self, dependent_coin):
        """
        Initialize the CryptocurrencyModelGenerator object.

        Parameters:
        dependent_coin (str): The dependent cryptocurrency symbol.
        """
        self.dependent_coin = dependent_coin
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d")
        self.working_dir = os.getenv("WORKING_DIR")
        self.df_stationary_tests = None
        self.df_price_data = None
        self.df_symbols = None
        self.df_pivot = None
        self.crypto_pairs = None
        self.df_all = None
        self.df_stationary_tests_resid = None
        self.df_final = None
        self.models_to_make = None
        self.models = None
        self.df_models = None
        self.conn = sqlite3.connect("cryptocurrency_data.db")
        self.database = "./data/database.db"

        initiate_logger(
            log_location="./logs",
            log_name=os.path.basename(__file__).rstrip(".py"),
        )

    def query_stationary_test_results(self):
        """
        Query the results of the stationary tests from a SQLite database.
        """
        logging.info("Querying stationary test results...")
        self.df_stationary_tests = query_stationary_test()

    def filter_stationary_test_results(self):
        """
        Filter the results of the stationary tests to keep only the symbols that passed the tests.
        """
        logging.info("Filtering stationary test results...")
        fk_symbols = filter_adf_pp_kpss(self.df_stationary_tests)

        with sqlite3.connect(self.database) as conn:
            self.df_symbols = pd.read_sql_query(
                "select pk_symbols as fk_symbols, id from DimSymbols", conn
            )

        self.df_symbols = self.df_symbols.merge(
            pd.DataFrame(fk_symbols, columns=["fk_symbols"]),
            on="fk_symbols",
            how="inner",
        )

    def query_top_price_data(self):
        """
        Query the top 100 price data records from a SQLite database.
        """
        logging.info("Querying top price data...")
        self.df_price_data = query_topN_price_data(100)
        self.df_price_data = pdc.downcast(self.df_price_data)

    def generate_crypto_pairs(self):
        """
        Generate all possible pairs of symbols from the list of symbols.
        """
        logging.info("Generating cryptocurrency pairs...")
        all_symbols = self.df_symbols.id.unique()
        self.crypto_pairs = create_models.generate_crypto_pairs(
            dependent_coin=self.dependent_coin,
            all_symbols=all_symbols,
            min_pairs=2,
            max_pairs=3,
        )

    def create_pivot_table(self):
        """
        Create a pivot table of the price data.
        """
        logging.info("Creating pivot table...")
        self.df_pivot = self.df_price_data.pivot_table(
            values="close_price",
            index=["date"],
            columns="id",
        ).reset_index()

    def calculate_residuals(self):
        """
        Calculate the residuals for each pair of symbols.
        """
        logging.info("Calculating residuals...")
        residuals = []
        for df in self.iter_list:
            df_tmp = calc_resid(df)
            residuals.append(df_tmp)
        self.df_all = pd.concat(residuals)

    def run_stationary_tests(self):
        """
        Run stationary tests on the residuals.
        """
        logging.info("Running stationary tests...")
        self.df_stationary_tests_resid = adf_pp_kpss(self.df_all)

    def filter_stationary_tests_resid(self):
        """
        Filter the results of the stationary tests on the residuals to keep only the pairs that passed the tests.
        """
        logging.info("Filtering stationary test results on residuals...")
        conditions = [
            (
                (self.df_stationary_tests_resid.PValue < 0.05)
                & (self.df_stationary_tests_resid.TestName == "Augmented Dickey-Fuller")
            ),
            (
                (self.df_stationary_tests_resid.PValue < 0.05)
                & (self.df_stationary_tests_resid.TestName == "Phillips-Perron Test")
            ),
            (
                (self.df_stationary_tests_resid.PValue > 0.1)
                & (self.df_stationary_tests_resid.TestName == "KPSS Stationarity Test")
            ),
        ]
        choices = [1, 1, 1]
        self.df_stationary_tests_resid["Keep"] = np.select(
            conditions, choices, default=0
        )
        self.df_final = (
            self.df_stationary_tests_resid.groupby(["TimeSeriesName"])["Keep"]
            .sum()
            .reset_index()
        )

    def extract_models_to_make(self):
        """
        Extract the list of pairs of symbols to use for generating the linear regression models.
        """
        logging.info("Extracting models to make...")
        self.models_to_make = self.df_final.TimeSeriesName.apply(
            lambda row: row.split("_")[:-1]
        ).to_list()

    def generate_linear_regression_models(self):
        """
        Generate the linear regression models for each pair of symbols.
        """
        logging.info("Generating linear regression models...")
        self.models = []
        for symbol_list in tqdm.tqdm(self.models_to_make):
            model_output = create_models.linear_model(
                df=self.df_pivot[symbol_list], symbol_list=symbol_list
            )
            self.models.append(model_output)

    def concatenate_models(self):
        """
        Concatenate the linear regression models into a single dataframe.
        """
        logging.info("Concatenating linear regression models...")
        self.df_models = pd.concat(self.models)
        self.df_models = self.df_models.reset_index(drop=True)

    def save_models_to_file(self):
        """
        Save the linear regression models to a parquet file.
        """
        logging.info("Saving linear regression models to file...")
        self.df_models.to_parquet(
            f"./data/{self.timestamp}_{self.dependent_coin}_models.parquet"
        )

    def save_models_to_database(self):
        """
        Save the linear regression models to a table in the database.
        """
        logging.info("Saving linear regression models to database...")
        with sqlite3.connect(self.database) as conn:
            # Write models to database
            self.df_models.to_sql("tblModels", conn, if_exists="append", index=False)

    def generate_iter_list_parallel(self):
        """
        Generate the iter_list of dataframes in parallel.
        """
        logging.info("Generating iter_list in parallel...")
        self.iter_list = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for crypto_pair in self.crypto_pairs:
                cols2keep = ["date"] + list(crypto_pair)
                futures.append(executor.submit(self.generate_df_tmp, cols2keep))
            for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
                df_tmp = future.result()
                if len(df_tmp) > 200:
                    self.iter_list.append(df_tmp)

    def generate_df_tmp(self, cols2keep):
        """
        Generate a df_tmp dataframe for a given set of columns.
        """
        df_tmp = self.df_pivot[cols2keep].dropna()
        return df_tmp

    def run(self):
        """
        Run the entire process of generating the linear regression models.
        """
        try:
            logging.info("Starting model generation process...")
            self.query_stationary_test_results()
            self.filter_stationary_test_results()
            self.query_top_price_data()
            self.generate_crypto_pairs()
            self.create_pivot_table()
            self.generate_iter_list_parallel()
            self.calculate_residuals()
            self.run_stationary_tests()
            self.filter_stationary_tests_resid()
            self.extract_models_to_make()
            self.generate_linear_regression_models()
            self.concatenate_models()
            self.save_models_to_database()
            # self.save_models_to_file()
            logging.info("Model generation process complete.")
        except Exception as e:
            logging.exception("An unexpected error occurred: {}".format(str(e)))


if __name__ == "__main__":
    model_generator = CryptocurrencyModelGenerator(dependent_coin="BTCUSDT")
    # model_generator = CryptocurrencyModelGenerator(dependent_coin="ETHUSDT")
    model_generator.run()
