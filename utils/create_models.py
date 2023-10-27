import pandas as pd
import numpy as np
import pandas as pd
import datetime
import statsmodels.api as sm
import numpy as np
import tqdm
import itertools
import requests
import json
import multiprocessing
import warnings


def generate_crypto_pairs(dependent_coin, all_symbols, min_pairs=2, max_pairs=3):
    crypto_pairs = []
    for coin in tqdm.tqdm(all_symbols):
        for i in range(min_pairs - 1, max_pairs):
            if coin in [dependent_coin]:
                symbols2combine = [s for s in all_symbols if s != coin]
                combs = list(itertools.combinations(symbols2combine, i))
                crypto_pairs = crypto_pairs + [c + (coin,) for c in combs]

    return crypto_pairs


def linear_model(df, symbol_list):
    df = df.dropna()
    training_from_date = df.index.min()
    training_to_date = df.index.max()

    try:
        X = sm.add_constant(df[symbol_list[:-1]])
        y = df[symbol_list[-1]]
        model = sm.OLS(y, X).fit()
        df_coeff = model.params.reset_index()
        df_coeff.columns = ["symbol", "coeff"]
        df_coeff.loc[df_coeff.symbol == "const", "coeff"] = 1
        df_coeff.loc[df_coeff.symbol == "const", "symbol"] = symbol_list[-1]
        # orient : str {'dict', 'list', 'series', 'split', 'tight', 'records', 'index'}
        coeff = json.dumps(df_coeff.set_index("symbol").to_dict("dict"))

        data = {
            "model_name": "_".join(symbol_list),
            "model_coeff": coeff,
            "model_date": datetime.datetime.now(),
            "model_type": "linear regression",
            "training_from_date": training_from_date,
            "training_to_date": training_to_date,
        }

        df_out = pd.DataFrame(data, index=[0])

    except Exception:
        logging.exception(f"Error generating linear model for {symbol_list}")
        df_out = pd.DataFrame()

    return df_out
