# Import libraries
import pandas as pd
import ccxt
import logging
import datetime
import tqdm
import os
import sqlite3
from dotenv import load_dotenv

# Load environment variables
_ = load_dotenv()
working_dir = os.getenv("WORKING_DIR")

# Change to working dir
os.chdir(working_dir)

# Import custom modules
from utils.custom_logger import initiate_logger
from utils.database_tools import (
    query_topN_symbols,
    query_topN_symbols_with_last_date,
)

initiate_logger(
    log_location="./logs",
    log_name=os.path.basename(__file__).rstrip(".py"),
)


def fetch_historical_data(row, exchange, timeframe="1d", from_date=None, limit=1000):
    # Initialize the exchange
    symbol = row.symbol
    pk_symbols = row.fk_symbols

    exchange = ccxt.binance(
        {
            "rateLimit": 2000,
            "enableRateLimit": True,
            "verbose": False,
            #'apiKey': API_KEY,
            #'secret': API_SECRET,
        }
    )

    exchange.load_markets()

    if from_date:
        since = int(from_date.timestamp() * 1e3)
    else:
        since = None

    # Fetch the historical data
    try:
        historical_data = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        df = pd.DataFrame(
            historical_data,
            columns=[
                "timestamp",
                "open_price",
                "high_price",
                "low_price",
                "close_price",
                "volume",
            ],
        )
        df = df.assign(
            date=pd.to_datetime(df.timestamp, unit="ms"),
            symbol=symbol,
            timeframe=timeframe,
            download_date=datetime.datetime.now(),
            fk_symbols=pk_symbols,
        )
    except (ccxt.RequestTimeout, ccxt.ExchangeError) as e:
        logging.exception(
            "RequestTimeout or ExchangeError: Failed to fetch historical data",
            exc_info=True,
        )
        df = pd.DataFrame()
    except Exception as e:
        logging.exception(
            "Unexpected Error: Failed to fetch historical data", exc_info=True
        )
        df = pd.DataFrame()
    finally:
        return df


if __name__ == "__main__":
    database = "./data/database.db"
    exchange = "binance"
    timeframe = "1d"
    incremental_download = False
    data_list = []
    topN = 100
    
    if incremental_download:
        df_symbols = query_topN_symbols_with_last_date(topN)
    else:
        df_symbols = query_topN_symbols_with_last_date(topN)
        #df_symbols = df_symbols[df_symbols.cmcRank <= 100]
        df_symbols["start_date"] = datetime.datetime(2018, 1, 1)

    # Iter rows over symbols with tqdm
    for _, row in tqdm.tqdm(df_symbols.iterrows(), total=df_symbols.shape[0]):
        from_date = row.start_date
        #end_date = datetime.date.today() - datetime.timedelta(days=1)
        end_date = row.first_date
        
        dfs = []
        while True:
            # Get data
            logging.info(
                f'Collecting data for {row.symbol} from {from_date.strftime("%Y-%m-%d %H:%M")}'
            )
            df = fetch_historical_data(row, exchange, timeframe, from_date)
            dfs.append(df)
            # Check results
            if df.empty:
                break 
            else:
                # Assign last date to from_date
                from_date = df.iloc[-1].date + datetime.timedelta(hours=1)

            if from_date >= end_date:
                # This checks if the from_date has reached the end_date
                break
            
        df_all = pd.concat(dfs)

        # Write data to database
        with sqlite3.connect(database) as conn:
            logging.info("Writing to database")
            df_all.to_sql("FactPriceData", conn, if_exists="append", index=False)

    # df_all = pd.concat(data_list)
    # file_name = f"all_{exchange}_{timeframe}.csv"
    # df_all.to_csv(os.path.join("./data", file_name), index=False)
