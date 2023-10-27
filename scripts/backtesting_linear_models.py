# Import libraries
import os
import pandas as pd
import tqdm
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from dotenv import load_dotenv
import warnings
import concurrent.futures

# Load environment variables
_ = load_dotenv()
working_dir = os.getenv("WORKING_DIR")

# Change to working dir
os.chdir(working_dir)

# Load custom libraries
from utils.custom_logger import initiate_logger
from utils.database_tools import (
    query_topN_price_data,
    query_latest_models,
)

# Suppress all warning messages
warnings.filterwarnings("ignore")

# Define global variable
std_threshold = 1.0


class BacktestingSpread:
    def __init__(self, database, topN=200, leverage=1):
        self.database = database
        self.topN = topN
        self.leverage = leverage
        self.df_models = None
        self.df_coins = None
        self.output_list = []

    def load_data(self):
        # Load models
        self.df_models = query_latest_models()

        # Query price data
        self.df_coins = query_topN_price_data(self.topN)

    def prepare_ohlc_data_with_spread(self, model):
        values = ["open_price", "high_price", "low_price", "close_price"]
        model_name = model.model_name
        symbol_list = model_name.split("_")
        model_coeff = {k: v for k, v in model.model_coeff.items() if v is not None}
        df_coeff = (
            pd.DataFrame(model_coeff["coeff"], index=[0])
            .T.reset_index()
            .rename(columns={"index": "id", 0: "coeff"})
        )

        df_tmp_coins = self.df_coins.loc[self.df_coins.id.isin(symbol_list)].copy()
        df_tmp_coins = df_tmp_coins.drop(
            columns=["symbol", "symbol_short", "fk_symbols", "volume"]
        )

        df_tmp_coins = df_tmp_coins.merge(df_coeff, on="id", how="left")

        # Calculate spread values
        df_input = pd.DataFrame()
        for value in values:
            df_pivot = df_tmp_coins.pivot_table(
                values=value,
                index=["date"],
                columns="id",
            ).reset_index()

            df = df_pivot[["date"] + symbol_list]
            df = df.dropna()
            df[value] = 0

            for symbol in symbol_list:
                df[value] = df[value] + df[symbol] * model_coeff["coeff"][symbol]

            if value == values[0]:
                df_input = df[["date", value]]
            else:
                df_input = pd.concat([df_input, df[value]], axis=1)

        df_input = df_input.assign(
            date=pd.to_datetime(df_input.date),
            spread=df_input.close_price,
            mean=df_input.close_price.mean(),
            std=df_input.close_price.std(),
            id=model_name,
            coeff=1,
        )

        df_final = pd.merge(
            df_input[["date", "spread", "mean", "std"]],
            df_tmp_coins,
            on="date",
            how="inner",
        )

        df_final = pd.concat([df_final, df_input])

        df_final = df_final.assign(model_name=model_name)

        col_sort = [
            "model_name",
            "id",
            "date",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "spread",
            "mean",
            "std",
            "coeff",
        ]
        df_final = df_final[col_sort].rename(
            columns={
                "open_price": "Open",
                "high_price": "High",
                "low_price": "Low",
                "close_price": "Close",
                "spread": "Spread",
                "mean": "Mean",
                "std": "Std",
                "coeff": "Coeff",
            }
        )

        return df_final

    def run_backtest(self, model):
        df_final = self.prepare_ohlc_data_with_spread(model)
        unique_ids = df_final.id.unique().tolist()
        model_name = df_final.model_name.unique().tolist()[0]

        for idx in unique_ids:
            df_tmp = df_final.loc[df_final.id == idx].copy()

            df_tmp = df_tmp.set_index("date")

            # Run backtest
            bt = Backtest(
                df_tmp,
                Spread,
                cash=1000000,
                commission=0.002,
                exclusive_orders=False,
                margin=1,  # 1 / self.leverage,  # Margin = 1 / leverage
            )

            output = bt.run()
            df_output = output.to_frame().T
            df_output["id"] = idx
            df_output["model_name"] = model_name

            self.output_list.append(df_output)

    def run(self):
        self.load_data()
        # Create pivot table
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for _, model in tqdm.tqdm(
                self.df_models.iterrows(), total=self.df_models.shape[0]
            ):
                try:
                    futures.append(executor.submit(self.run_backtest, model))
                except Exception as e:
                    logging.error(f"Error running backtest for model {model_name}: {e}")
                    continue

            # Wait for all futures to complete
            concurrent.futures.wait(futures)

        # Save backtesting results to file
        df_all_output = pd.concat(self.output_list)
        output_file = f"./data/backtesting_spread_15std_{self.leverage}xLeverage.xlsx"
        df_all_output.to_excel(output_file)
        # logging.info(f"Backtesting results saved to {output_file}")


class Spread(Strategy):
    def init(self):
        pass

    def next(self):
        # Calculate buy and sell signals based on mean and standard deviation
        if crossover(self.data.Spread, self.data.Mean - self.data.Std * std_threshold):
            if self.data.Coeff >= 0:
                self.buy(size=1)
            elif self.data.Coeff < 0:
                self.sell(size=1)
        elif crossover(
            self.data.Spread, self.data.Mean + self.data.Std * std_threshold
        ):
            if self.data.Coeff >= 0:
                self.sell(size=1)
            elif self.data.Coeff < 0:
                self.buy(size=1)
        elif crossover(self.data.Spread, self.data.Mean):
            self.position.close()


if __name__ == "__main__":
    models_file = "./data/models.parquet"
    database = "./data/database.db"
    topN = 200
    leverage = 5

    backtesting = BacktestingSpread(database, topN, leverage)
    backtesting.run()
